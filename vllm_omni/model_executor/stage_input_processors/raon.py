# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Raon Stage-0 -> Stage-1 bridge."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.transformers_utils.configs.raon import ENV
from vllm_omni.model_executor.models.raon.raon_utils import (
    collapse_exact_repeated_codec_snapshot,
    unwrap_singleton_list,
)

logger = init_logger(__name__)

_CODEC_CHUNK_KEY = "codec_codes_chunk"
_CODEC_FULL_KEY = "codec_codes"
_MAX_CODEC_GROUPS = 32

_REQUEST_CODEC_CHUNKS: dict[str, list[torch.Tensor]] = defaultdict(list)


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> list[Any]:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    source_stage = stage_list[source_stage_id]
    if source_stage.engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    return source_stage.engine_outputs


def _extract_codec_payload(codec_payload: Any) -> Any:
    codec_payload = unwrap_singleton_list(codec_payload)
    if isinstance(codec_payload, dict):
        codec_payload = codec_payload.get(
            _CODEC_CHUNK_KEY,
            codec_payload.get(_CODEC_FULL_KEY, codec_payload.get("codes")),
        )
        codec_payload = unwrap_singleton_list(codec_payload)
    return codec_payload


def _to_long_tensor(payload: Any) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload.to(torch.long)
    return torch.as_tensor(payload, dtype=torch.long)


def _flatten_codec_codes(codec_payload: Any) -> list[int]:
    codec_payload = _extract_codec_payload(codec_payload)
    if codec_payload is None:
        return []

    codes = _to_long_tensor(codec_payload)
    if codes.numel() == 0:
        return []

    if codes.ndim == 1:
        return codes.reshape(-1).tolist()

    if codes.ndim == 2:
        if 1 < codes.shape[-1] <= _MAX_CODEC_GROUPS:
            return codes.reshape(-1).tolist()
        if 1 < codes.shape[0] <= _MAX_CODEC_GROUPS:
            return codes.transpose(0, 1).reshape(-1).tolist()
        return codes.reshape(-1).tolist()

    if codes.ndim == 3:
        if 1 < codes.shape[-1] <= _MAX_CODEC_GROUPS:
            merged = codes.reshape(-1, codes.shape[-1])
            return merged.reshape(-1).tolist()
        if 1 < codes.shape[1] <= _MAX_CODEC_GROUPS:
            merged = codes.transpose(1, 2).reshape(-1, codes.shape[1])
            return merged.reshape(-1).tolist()
        return codes.reshape(-1).tolist()

    if codes.ndim > 3:
        return _flatten_codec_codes(codes.reshape(codes.shape[0], -1, codes.shape[-1]))

    return []


def _codec_payload_to_time_major(codec_payload: Any) -> torch.Tensor | None:
    """Normalize payload into [T, G] time-major codec codes when possible."""
    codec_payload = _extract_codec_payload(codec_payload)
    if codec_payload is None:
        return None

    if isinstance(codec_payload, (list, tuple)):
        normalized_items: list[torch.Tensor] = []
        for item in codec_payload:
            normalized = _codec_payload_to_time_major(item)
            if isinstance(normalized, torch.Tensor) and normalized.numel() > 0:
                normalized_items.append(normalized.to(torch.long))

        if len(normalized_items) == 0:
            return None
        if len(normalized_items) == 1:
            return normalized_items[0]

        same_shape = all(t.shape == normalized_items[0].shape for t in normalized_items[1:])
        if same_shape:
            reference = normalized_items[-1]
            if (
                int(reference.shape[0]) > 1
                and int(reference.shape[1]) > 1
                and all(torch.equal(reference, t) for t in normalized_items[:-1])
            ):
                logger.warning(
                    "[raon stage0_to_stage1] collapsing duplicate full snapshots in list: count=%d shape=%s",
                    len(normalized_items),
                    tuple(reference.shape),
                )
                return reference

        same_groups = all(int(t.shape[1]) == int(normalized_items[0].shape[1]) for t in normalized_items)
        if same_groups:
            # Handle cumulative snapshots like [c1], [c1..c2], ..., [c1..cn].
            cumulative_prefix = True
            prev = normalized_items[0]
            for current in normalized_items[1:]:
                if int(current.shape[0]) < int(prev.shape[0]) or not torch.equal(current[: prev.shape[0]], prev):
                    cumulative_prefix = False
                    break
                prev = current
            if cumulative_prefix and int(normalized_items[-1].shape[0]) > int(normalized_items[0].shape[0]):
                logger.warning(
                    "[raon stage0_to_stage1] collapsing cumulative snapshots in list: count=%d final_shape=%s",
                    len(normalized_items),
                    tuple(normalized_items[-1].shape),
                )
                return normalized_items[-1]
            return torch.cat(normalized_items, dim=0)

        flattened = torch.cat([item.reshape(-1) for item in normalized_items], dim=0)
        return flattened.reshape(1, -1)

    codes = _to_long_tensor(codec_payload)

    # Stage-0 may surface a batch/list of *identical full snapshots* for a
    # single request (e.g., one copy per scheduled row in a decode tick). If
    # we flatten [N, T, G] directly, Stage-1 decodes T multiple times and the
    # utterance is repeated (commonly x3).
    if codes.ndim == 3 and int(codes.shape[0]) > 1 and int(codes.shape[1]) > 1 and int(codes.shape[2]) > 1:
        reference = codes[-1]
        if all(torch.equal(reference, codes[i]) for i in range(int(codes.shape[0]) - 1)):
            logger.warning(
                "[raon stage0_to_stage1] collapsing duplicate full-snapshot batch: shape=%s -> %s",
                tuple(codes.shape),
                tuple(reference.shape),
            )
            codes = reference

    if codes.numel() == 0:
        return None

    if codes.ndim == 0:
        return codes.reshape(1, 1)

    if codes.ndim == 1:
        return codes.reshape(1, -1)

    if codes.ndim == 2:
        if 1 < codes.shape[-1] <= _MAX_CODEC_GROUPS:
            return codes
        if 1 < codes.shape[0] <= _MAX_CODEC_GROUPS:
            return codes.transpose(0, 1)
        return codes.reshape(1, -1)

    if codes.ndim == 3:
        if 1 < codes.shape[-1] <= _MAX_CODEC_GROUPS:
            return codes.reshape(-1, codes.shape[-1])
        if 1 < codes.shape[1] <= _MAX_CODEC_GROUPS:
            return codes.transpose(1, 2).reshape(-1, codes.shape[1])
        return codes.reshape(1, -1)

    if codes.ndim > 3:
        return _codec_payload_to_time_major(codes.reshape(codes.shape[0], -1, codes.shape[-1]))

    return None


def _resolve_request_id(stage0_output: Any, default_idx: int) -> str:
    for attr in ("request_id", "req_id", "id"):
        req_id = getattr(stage0_output, attr, None)
        if req_id is not None:
            req_str = str(req_id)
            if req_str:
                return req_str

    outputs = getattr(stage0_output, "outputs", None)
    if outputs:
        first_output = outputs[0]
        for attr in ("request_id", "req_id", "id"):
            req_id = getattr(first_output, attr, None)
            if req_id is not None:
                req_str = str(req_id)
                if req_str:
                    return req_str

    return f"req-{default_idx}"


def _append_request_chunk(req_id: str, chunk_time_major: torch.Tensor) -> None:
    chunk_cpu = chunk_time_major.to("cpu").contiguous()
    existing = _REQUEST_CODEC_CHUNKS[req_id]
    if not existing:
        existing.append(chunk_cpu)
        return

    prev = torch.cat(existing, dim=0)
    prev_flat = prev.reshape(-1)
    chunk_flat = chunk_cpu.reshape(-1)
    if (
        chunk_cpu.shape[0] >= prev.shape[0]
        and chunk_cpu.shape[1] == prev.shape[1]
        and torch.equal(chunk_cpu[: prev.shape[0]], prev)
    ):
        _REQUEST_CODEC_CHUNKS[req_id] = [chunk_cpu]
        return
    if chunk_flat.shape[0] >= prev_flat.shape[0] and torch.equal(chunk_flat[: prev_flat.shape[0]], prev_flat):
        _REQUEST_CODEC_CHUNKS[req_id] = [chunk_cpu]
        return

    if prev_flat.shape[0] >= chunk_flat.shape[0] and torch.equal(prev_flat[: chunk_flat.shape[0]], chunk_flat):
        return

    existing.append(chunk_cpu)


def _collapse_cumulative_prefix_snapshot(payload_time_major: torch.Tensor) -> torch.Tensor:
    """Collapse [c1] + [c1,c2] + ... style snapshots to the final window."""
    if payload_time_major.ndim != 2 or payload_time_major.shape[0] < 3:
        return payload_time_major

    first_row = payload_time_major[0]
    starts = torch.nonzero((payload_time_major == first_row).all(dim=1), as_tuple=False).flatten()
    if starts.numel() < 3:
        return payload_time_major

    last_start = int(starts[-1].item())
    if last_start <= 0 or last_start >= int(payload_time_major.shape[0]):
        return payload_time_major

    candidate = payload_time_major[last_start:]
    if candidate.numel() == 0:
        return payload_time_major

    prev_start = int(starts[-2].item())
    prev_snapshot_len = max(0, last_start - prev_start)
    overlap = min(int(candidate.shape[0]), prev_snapshot_len)
    if overlap <= 0:
        return payload_time_major
    if not torch.equal(payload_time_major[prev_start : prev_start + overlap], candidate[:overlap]):
        return payload_time_major

    logger.warning(
        "[raon stage0_to_stage1] collapsed cumulative snapshot payload: original_shape=%s -> collapsed_shape=%s",
        tuple(payload_time_major.shape),
        tuple(candidate.shape),
    )
    return candidate


def _resolve_mm_output(stage0_output: Any, first_output: Any) -> dict[str, Any] | None:
    payload_candidates = (
        getattr(first_output, "multimodal_output", None) if first_output is not None else None,
        getattr(stage0_output, "multimodal_output", None),
        getattr(first_output, "pooling_output", None) if first_output is not None else None,
        getattr(stage0_output, "pooling_output", None),
    )
    for payload in payload_candidates:
        if isinstance(payload, dict):
            return payload
    return None


def _is_stage0_request_finished(stage0_output: Any, first_output: Any) -> bool:
    if bool(getattr(stage0_output, "finished", False)):
        return True
    if first_output is None:
        return False
    return getattr(first_output, "finish_reason", None) is not None


def stage0_to_stage1(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build Stage-1 prompts from Stage-0 codec payloads."""
    # Read continuation_silence_frames from incoming prompt before discarding
    csf = 0
    if isinstance(prompt, dict):
        src_add_info = prompt.get("additional_information")
        if isinstance(src_add_info, dict):
            csf_raw = src_add_info.get("continuation_silence_frames")
            if isinstance(csf_raw, list) and len(csf_raw) > 0:
                csf = int(csf_raw[0])
            elif isinstance(csf_raw, (int, float)):
                csf = int(csf_raw)
    del prompt, requires_multimodal_data

    stage0_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    if len(stage0_outputs) == 0:
        return []

    stage1_prompts: list[OmniTokensPrompt] = []

    for out_idx, stage0_output in enumerate(stage0_outputs):
        outputs = getattr(stage0_output, "outputs", None)
        first_output = outputs[0] if outputs else None
        req_id = _resolve_request_id(stage0_output, out_idx)

        mm_output = _resolve_mm_output(stage0_output, first_output)
        full_time_major: torch.Tensor | None = None
        if isinstance(mm_output, dict):
            chunk_payload = mm_output.get(_CODEC_CHUNK_KEY)
            if chunk_payload is not None:
                time_major = _codec_payload_to_time_major(chunk_payload)
                if time_major is not None and time_major.numel() > 0:
                    _append_request_chunk(req_id, time_major)

            full_payload = mm_output.get(_CODEC_FULL_KEY)
            if full_payload is not None:
                full_time_major = _codec_payload_to_time_major(full_payload)

        if not _is_stage0_request_finished(stage0_output, first_output):
            continue

        buffered_chunks = _REQUEST_CODEC_CHUNKS.pop(req_id, [])
        if len(buffered_chunks) > 0:
            all_codes_time_major = torch.cat(buffered_chunks, dim=0)
        else:
            if not (isinstance(full_time_major, torch.Tensor) and full_time_major.numel() > 0):
                continue
            all_codes_time_major = full_time_major
        all_codes_time_major = _collapse_cumulative_prefix_snapshot(all_codes_time_major)
        all_codes_time_major = collapse_exact_repeated_codec_snapshot(all_codes_time_major)
        all_flattened_codes = _flatten_codec_codes(all_codes_time_major)
        if len(all_flattened_codes) == 0:
            continue

        logger.info(
            "[raon stage0_to_stage1] req_id=%s chunks=%d, time_major_shape=%s, flattened_len=%d",
            req_id,
            len(buffered_chunks),
            tuple(all_codes_time_major.shape),
            len(all_flattened_codes),
        )

        stage1_prompts.append(
            OmniTokensPrompt(
                prompt_token_ids=(
                    all_flattened_codes if len(all_flattened_codes) <= ENV.stage1_max_prompt_tokens else [0]
                ),
                additional_information={
                    "codec_codes": all_codes_time_major.to(torch.long).contiguous(),
                    "_omni_req_id": [req_id],
                    "continuation_silence_frames": csf,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return stage1_prompts


_DEFAULT_CHUNK_FRAMES: int = ENV.async_chunk_interval
_ASYNC_ALL_CODES: dict[str, list[torch.Tensor]] = defaultdict(list)
_ASYNC_EMITTED_FRAMES: dict[str, int] = defaultdict(int)
_ASYNC_REQ_ID_MAP: dict[str, str] = {}


def _get_connector_chunk_config(
    transfer_manager: Any,
) -> tuple[int, int]:
    """Read codec_chunk_frames / codec_left_context_frames from connector cfg.

    Falls back to ``ENV.async_chunk_interval`` (``RAON_ASYNC_CHUNK_INTERVAL``)
    for chunk size and 0 for left context when not set in YAML.
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_frames = int(cfg.get("codec_chunk_frames", 0))
    left_context_frames = int(cfg.get("codec_left_context_frames", 0))
    if chunk_frames <= 0:
        chunk_frames = _DEFAULT_CHUNK_FRAMES
    if left_context_frames < 0:
        left_context_frames = 0
    return chunk_frames, left_context_frames


def async_chunk_cleanup_request(request_id: str) -> None:
    """Clear async chunk state for either internal or external request ID."""
    _ASYNC_ALL_CODES.pop(request_id, None)
    _ASYNC_EMITTED_FRAMES.pop(request_id, None)
    mapped = _ASYNC_REQ_ID_MAP.pop(request_id, None)
    if mapped and mapped != request_id:
        _ASYNC_ALL_CODES.pop(mapped, None)
        _ASYNC_EMITTED_FRAMES.pop(mapped, None)
    stale = [k for k, v in _ASYNC_REQ_ID_MAP.items() if v == request_id]
    for k in stale:
        _ASYNC_REQ_ID_MAP.pop(k, None)


def _cleanup_async_chunk_state(request_id: str, internal_id: str | None = None) -> None:
    _ASYNC_ALL_CODES.pop(request_id, None)
    _ASYNC_EMITTED_FRAMES.pop(request_id, None)
    if internal_id and internal_id != request_id:
        _ASYNC_REQ_ID_MAP.pop(internal_id, None)


def stage0_to_stage1_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Emit windowed Stage-0 codec chunks with left context for Stage-1.

    Each emitted window contains ``left_context + new_frames`` time-major
    codec codes.  Stage-1 decodes the full window through Mimi and trims
    the audio corresponding to the left-context prefix, producing smooth
    chunk boundaries.
    """
    request_id = request.external_req_id
    internal_id = getattr(request, "request_id", None)
    if internal_id and internal_id != request_id:
        _ASYNC_REQ_ID_MAP[internal_id] = request_id
    request_finished = is_finished or request.is_finished()

    chunk_size, left_context_cfg = _get_connector_chunk_config(transfer_manager)

    # Read continuation_silence_frames from additional_information
    csf = 0
    add_info = getattr(request, "additional_information", None)
    if isinstance(add_info, dict):
        csf_raw = add_info.get("continuation_silence_frames")
        if isinstance(csf_raw, list) and len(csf_raw) > 0:
            csf = int(csf_raw[0])
        elif isinstance(csf_raw, (int, float)):
            csf = int(csf_raw)

    # --- accumulate incoming codec payload ---
    _has_codec_keys = isinstance(pooling_output, dict) and (
        _CODEC_CHUNK_KEY in pooling_output or _CODEC_FULL_KEY in pooling_output
    )
    if not _has_codec_keys:
        if not _ASYNC_ALL_CODES.get(request_id) and _ASYNC_EMITTED_FRAMES.get(request_id, 0) == 0:
            if request_finished:
                _cleanup_async_chunk_state(request_id, internal_id)
                return {
                    "finished": torch.tensor(True, dtype=torch.bool),
                    "global_request_id": request_id,
                    "flush_only": True,
                }
            return None

    chunk_payload = None
    if pooling_output is not None:
        chunk_payload = pooling_output.get(_CODEC_CHUNK_KEY)
        if chunk_payload is None:
            chunk_payload = pooling_output.get(_CODEC_FULL_KEY)

    if chunk_payload is not None:
        time_major = _codec_payload_to_time_major(chunk_payload)
        if time_major is not None and time_major.numel() > 0:
            _ASYNC_ALL_CODES[request_id].append(time_major.to(torch.long).cpu().contiguous())

    # --- compute total accumulated frame count ---
    all_chunks = _ASYNC_ALL_CODES.get(request_id, [])
    total_frames = sum(int(c.shape[0]) for c in all_chunks) if all_chunks else 0
    emitted = _ASYNC_EMITTED_FRAMES.get(request_id, 0)
    pending = total_frames - emitted

    if total_frames == 0:
        if request_finished:
            _cleanup_async_chunk_state(request_id, internal_id)
            return {
                "finished": torch.tensor(True, dtype=torch.bool),
                "global_request_id": request_id,
                "flush_only": True,
            }
        return None

    if not request_finished and pending < chunk_size:
        return None

    # --- build windowed chunk with left context ---
    new_frames = min(pending, chunk_size)
    left_context = min(emitted, left_context_cfg)

    all_codes = torch.cat(all_chunks, dim=0)
    window_start = emitted - left_context
    window_end = emitted + new_frames
    window = all_codes[window_start:window_end]

    _ASYNC_EMITTED_FRAMES[request_id] = emitted + new_frames

    flattened = _flatten_codec_codes(window)

    payload: dict[str, Any] = {
        "codec_codes": flattened,
        "codec_codes_flat": flattened,
        "code_predictor_codes": flattened,
        "global_request_id": request_id,
        "finished": torch.tensor(request_finished, dtype=torch.bool),
        "flush_only": False,
        "left_context_size": left_context,
        "continuation_silence_frames": csf if emitted == 0 else 0,
    }

    if request_finished:
        _cleanup_async_chunk_state(request_id, internal_id)

    logger.info(
        "[raon stage0_to_stage1_async_chunk] req_id=%s total=%d emitted=%d "
        "new=%d left_ctx=%d finished=%s window_shape=%s",
        request_id,
        total_frames,
        emitted,
        new_frames,
        left_context,
        request_finished,
        tuple(window.shape),
    )

    return payload
