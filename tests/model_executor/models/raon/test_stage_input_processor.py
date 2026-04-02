# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.model_executor.stage_input_processors import raon as stage_proc


@dataclass
class _DummyChoice:
    multimodal_output: dict
    finish_reason: str | None = None


@dataclass
class _DummyReqOutput:
    request_id: str
    outputs: list[_DummyChoice]


@dataclass
class _DummyStage:
    engine_outputs: list[_DummyReqOutput]


def _build_stage(
    req_id: str, codes, *, finish_reason: str | None, chunk_key: str = "codec_codes"
) -> list[_DummyStage]:
    return [
        _DummyStage(
            engine_outputs=[
                _DummyReqOutput(
                    request_id=req_id,
                    outputs=[
                        _DummyChoice(
                            multimodal_output={chunk_key: codes},
                            finish_reason=finish_reason,
                        )
                    ],
                )
            ]
        )
    ]


def test_stage0_to_stage1_accumulates_chunks_until_finish():
    stage_proc._REQUEST_CODEC_CHUNKS.clear()
    try:
        # First tick: incremental chunk, request not finished yet.
        stage_list = _build_stage(
            "req-a", [[1, 2, 3], [4, 5, 6]], finish_reason=None, chunk_key="codec_codes_chunk"
        )
        prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
        assert prompts == []

        # Final tick: append final chunk and flush to Stage-1.
        stage_list = _build_stage("req-a", [[7, 8, 9]], finish_reason="stop", chunk_key="codec_codes_chunk")
        prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
        assert len(prompts) == 1
        assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    finally:
        stage_proc._REQUEST_CODEC_CHUNKS.clear()


def test_stage0_to_stage1_handles_cumulative_snapshot_without_duplication():
    stage_proc._REQUEST_CODEC_CHUNKS.clear()
    try:
        # First tick: initial chunk.
        stage_list = _build_stage("req-b", [[10, 11, 12]], finish_reason=None, chunk_key="codec_codes_chunk")
        prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
        assert prompts == []

        # Final tick emits cumulative [old + new]. The processor should
        # replace the buffer (not append duplicate prefix).
        stage_list = _build_stage(
            "req-b", [[10, 11, 12], [13, 14, 15]], finish_reason="stop", chunk_key="codec_codes_chunk"
        )
        prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
        assert len(prompts) == 1
        assert prompts[0]["prompt_token_ids"] == [10, 11, 12, 13, 14, 15]
    finally:
        stage_proc._REQUEST_CODEC_CHUNKS.clear()


def test_stage0_to_stage1_dedups_cumulative_snapshot_with_shape_variation():
    stage_proc._REQUEST_CODEC_CHUNKS.clear()
    try:
        # First tick in canonical [T, G].
        stage_list = _build_stage("req-c", [[1, 2], [3, 4]], finish_reason=None, chunk_key="codec_codes_chunk")
        prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
        assert prompts == []

        # Final tick arrives as flattened cumulative snapshot.
        stage_list = _build_stage(
            "req-c", [1, 2, 3, 4, 5, 6], finish_reason="stop", chunk_key="codec_codes_chunk"
        )
        prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
        assert len(prompts) == 1
        assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]
    finally:
        stage_proc._REQUEST_CODEC_CHUNKS.clear()


def test_stage0_to_stage1_prefers_incremental_chunk_payload_when_both_present():
    stage_proc._REQUEST_CODEC_CHUNKS.clear()
    try:
        stage_list = [
            _DummyStage(
                engine_outputs=[
                    _DummyReqOutput(
                        request_id="req-d",
                        outputs=[
                            _DummyChoice(
                                multimodal_output={
                                    "codec_codes": [[1, 2, 3], [4, 5, 6]],
                                    "codec_codes_chunk": [[4, 5, 6]],
                                },
                                finish_reason="stop",
                            )
                        ],
                    )
                ]
            )
        ]
        prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
        assert len(prompts) == 1
        assert prompts[0]["prompt_token_ids"] == [4, 5, 6]
    finally:
        stage_proc._REQUEST_CODEC_CHUNKS.clear()
