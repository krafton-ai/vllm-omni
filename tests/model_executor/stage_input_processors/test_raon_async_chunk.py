# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Raon stage0_to_stage1_async_chunk and async_chunk_cleanup_request."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vllm_omni.model_executor.stage_input_processors.raon import (
    _ASYNC_ALL_CODES,
    _ASYNC_EMITTED_FRAMES,
    _ASYNC_REQ_ID_MAP,
    _DEFAULT_CHUNK_FRAMES,
    async_chunk_cleanup_request,
    stage0_to_stage1_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_G = 8
_CHUNK_SIZE = _DEFAULT_CHUNK_FRAMES


def _req(
    external_req_id: str,
    *,
    finished: bool = False,
    request_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        external_req_id=external_req_id,
        request_id=request_id or external_req_id,
        is_finished=lambda: finished,
    )


def _tm() -> SimpleNamespace:
    return SimpleNamespace()


def _tm_with_config(chunk_frames: int = 10, left_context_frames: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        connector=SimpleNamespace(
            config={"extra": {
                "codec_chunk_frames": chunk_frames,
                "codec_left_context_frames": left_context_frames,
            }}
        )
    )


def _codes(T: int, G: int = _G) -> torch.Tensor:
    return torch.arange(T * G, dtype=torch.long).reshape(T, G)


def _pooling(*, chunk: torch.Tensor | None = None, full: torch.Tensor | None = None) -> dict:
    out: dict = {}
    if chunk is not None:
        out["codec_codes_chunk"] = chunk
    if full is not None:
        out["codec_codes"] = full
    return out


def _cleanup(*req_ids: str) -> None:
    for rid in req_ids:
        _ASYNC_ALL_CODES.pop(rid, None)
        _ASYNC_EMITTED_FRAMES.pop(rid, None)
        stale = [k for k, v in list(_ASYNC_REQ_ID_MAP.items()) if v == rid or k == rid]
        for k in stale:
            _ASYNC_REQ_ID_MAP.pop(k, None)


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    for store in (_ASYNC_ALL_CODES, _ASYNC_EMITTED_FRAMES, _ASYNC_REQ_ID_MAP):
        store.clear()


# ===================================================================
# YAML config validation
# ===================================================================


def test_raon_stage_yaml_wires_codec_chunk_frames_25():
    root = Path(__file__).resolve().parents[3]
    path = root / "vllm_omni" / "model_executor" / "stage_configs" / "raon.yaml"
    data = yaml.safe_load(path.read_text())
    assert data.get("async_chunk") is True
    extra = data["runtime"]["connectors"]["connector_of_shared_memory"]["extra"]
    assert int(extra["codec_chunk_frames"]) == 25
    assert int(extra["codec_left_context_frames"]) == 25


# ===================================================================
# async_chunk_cleanup_request
# ===================================================================


class TestAsyncChunkCleanupRequest:
    def test_removes_accumulated_state(self):
        rid = "cleanup-acc"
        _ASYNC_ALL_CODES[rid] = [_codes(5)]
        _ASYNC_EMITTED_FRAMES[rid] = 10
        async_chunk_cleanup_request(rid)
        assert rid not in _ASYNC_ALL_CODES
        assert rid not in _ASYNC_EMITTED_FRAMES

    def test_idempotent_when_state_absent(self):
        async_chunk_cleanup_request("cleanup-absent-xyz")

    def test_cleans_internal_to_external_mapping(self):
        internal_id = "int-id-cleanup"
        external_id = "ext-id-cleanup"
        _ASYNC_REQ_ID_MAP[internal_id] = external_id
        _ASYNC_ALL_CODES[external_id] = [_codes(3)]
        _ASYNC_EMITTED_FRAMES[external_id] = 3

        async_chunk_cleanup_request(internal_id)
        assert internal_id not in _ASYNC_REQ_ID_MAP
        assert external_id not in _ASYNC_ALL_CODES
        assert external_id not in _ASYNC_EMITTED_FRAMES

        _ASYNC_REQ_ID_MAP["int-reverse"] = "ext-reverse"
        async_chunk_cleanup_request("ext-reverse")
        assert "int-reverse" not in _ASYNC_REQ_ID_MAP


# ===================================================================
# Skip path (non-audio)
# ===================================================================


@pytest.mark.parametrize(
    "pooling_output",
    [None, {"some_other_key": 123}],
    ids=["none-pooling", "no-codec-keys"],
)
def test_skip_path_returns_none(pooling_output):
    req = _req("skip-test")
    assert stage0_to_stage1_async_chunk(_tm(), pooling_output, req) is None


# ===================================================================
# Windowed chunk emission
# ===================================================================


class TestAsyncChunkWindowedEmission:
    def test_first_chunk_waits_then_emits_at_chunk_size(self):
        rid = "windowed"
        req = _req(rid)
        assert stage0_to_stage1_async_chunk(_tm(), _pooling(chunk=_codes(1)), req) is None

        for _ in range(_CHUNK_SIZE - 1):
            payload = stage0_to_stage1_async_chunk(_tm(), _pooling(chunk=_codes(1)), req)
        assert payload is not None
        assert "codec_codes" in payload
        assert payload["left_context_size"] == 0
        assert _ASYNC_EMITTED_FRAMES.get(rid, 0) == _CHUNK_SIZE

    def test_second_chunk_includes_left_context(self):
        rid = "windowed-ctx"
        tm = _tm_with_config(chunk_frames=5, left_context_frames=3)
        req = _req(rid)
        for _ in range(5):
            stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        for _ in range(4):
            stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        payload = stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        assert payload is not None
        assert payload["left_context_size"] == 3
        assert len(payload["codec_codes"]) == 8 * _G

    def test_history_retained_after_emission(self):
        rid = "windowed-hist"
        req = _req(rid)
        for _ in range(_CHUNK_SIZE):
            stage0_to_stage1_async_chunk(_tm(), _pooling(chunk=_codes(1)), req)
        all_codes = _ASYNC_ALL_CODES.get(rid, [])
        total = sum(int(c.shape[0]) for c in all_codes)
        assert total == _CHUNK_SIZE


# ===================================================================
# Finish / flush path
# ===================================================================


class TestAsyncChunkFinishFlush:
    def test_finish_emits_accumulated_data_and_cleans_state(self):
        rid = "finish-flush"
        req_running = _req(rid, finished=False)
        req_done = _req(rid, finished=True)
        stage0_to_stage1_async_chunk(_tm(), _pooling(chunk=_codes(1)), req_running)
        payload = stage0_to_stage1_async_chunk(
            _tm(), _pooling(chunk=_codes(2)), req_done, is_finished=True
        )
        assert payload is not None
        assert payload["finished"].item() is True
        assert payload["flush_only"] is False
        assert rid not in _ASYNC_ALL_CODES
        assert rid not in _ASYNC_EMITTED_FRAMES

    def test_finish_with_no_accumulated_data(self):
        rid = "finish-no-data"
        req = _req(rid, finished=True)
        payload = stage0_to_stage1_async_chunk(
            _tm(), _pooling(chunk=_codes(1)), req, is_finished=True
        )
        assert payload is not None

    def test_finish_with_none_pooling_and_no_prior_data(self):
        rid = "finish-none"
        req = _req(rid, finished=True)
        payload = stage0_to_stage1_async_chunk(_tm(), None, req, is_finished=True)
        if payload is not None:
            assert payload.get("flush_only") is True


# ===================================================================
# Full payload fallback
# ===================================================================


def test_full_payload_key_used_when_chunk_absent():
    rid = "full-payload"
    req = _req(rid)
    payload = stage0_to_stage1_async_chunk(
        _tm(), _pooling(full=_codes(_CHUNK_SIZE)), req
    )
    assert payload is not None
    assert "codec_codes" in payload


# ===================================================================
# Request ID mapping
# ===================================================================


class TestAsyncChunkReqIdMapping:
    def test_internal_id_mapped_and_cleanup_works(self):
        internal_id = "int-id-map"
        external_id = "ext-id-map"
        req = SimpleNamespace(
            external_req_id=external_id,
            request_id=internal_id,
            is_finished=lambda: False,
        )
        stage0_to_stage1_async_chunk(_tm(), _pooling(chunk=_codes(1)), req)
        assert _ASYNC_REQ_ID_MAP.get(internal_id) == external_id

        async_chunk_cleanup_request(internal_id)
        assert internal_id not in _ASYNC_REQ_ID_MAP
        assert external_id not in _ASYNC_ALL_CODES
        assert external_id not in _ASYNC_EMITTED_FRAMES


# ===================================================================
# Payload structure validation (comprehensive)
# ===================================================================


def test_payload_structure_comprehensive():
    rid = "payload-struct"
    req = _req(rid, finished=False)
    payload = stage0_to_stage1_async_chunk(
        _tm(), _pooling(chunk=_codes(_CHUNK_SIZE)), req
    )
    assert payload is not None

    for key in ("codec_codes", "codec_codes_flat", "global_request_id",
                 "finished", "flush_only", "left_context_size"):
        assert key in payload, f"missing key: {key}"

    assert isinstance(payload["finished"], torch.Tensor)
    assert payload["finished"].dtype == torch.bool
    assert payload["finished"].item() is False

    codes = payload["codec_codes"]
    assert isinstance(codes, list)
    assert all(isinstance(v, int) for v in codes)
    assert len(codes) == _CHUNK_SIZE * _G

    assert isinstance(payload["left_context_size"], int)


# ===================================================================
# _trim_left_context_audio
# ===================================================================


class TestTrimLeftContextAudio:
    @pytest.mark.parametrize(
        "audio_len,left_ctx,total_frames,expected_len",
        [
            (100, 0, 10, 100),
            (1000, 2, 10, 800),
            (100, 10, 10, 0),
            (100, 5, 0, 100),
        ],
        ids=["no-trim", "proportional", "exceeds-length", "zero-total-frames"],
    )
    def test_trim(self, audio_len, left_ctx, total_frames, expected_len):
        from vllm_omni.model_executor.models.raon.raon_code2wav import RaonCode2WavModel

        audio = torch.randn(audio_len)
        result = RaonCode2WavModel._trim_left_context_audio(audio, left_ctx, total_frames)
        assert result.shape[-1] == expected_len
