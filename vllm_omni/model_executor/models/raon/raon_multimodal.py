# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from vllm.logger import init_logger

from vllm_omni.tokenizers.raon_tokenizer import (
    ALL_AUDIO_PLACEHOLDER_VARIANTS,
    AUDIO_END,
    AUDIO_END_TOKEN,
    AUDIO_INPUT_PAD_TOKEN,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_OPEN_SEQ,
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PLACEHOLDER_SEQ,
    AUDIO_PLACEHOLDER_PATTERN,
    AUDIO_PLACEHOLDER_SEQ,
    AUDIO_START,
    AUDIO_START_TOKEN,
    USER_PROMPT_MARKER,
    align_tokenizer,
    inject_placeholders_into_str,
    inject_placeholders_into_token_ids,
)
from vllm_omni.transformers_utils.configs.raon import get_mimi_frame_rate

logger = init_logger(__name__)

LEGACY_AUDIO_PAD_TOKEN_ID = 151673
LEGACY_AUDIO_PLACEHOLDER_TOKEN_IDS = [
    AUDIO_START.id,
    LEGACY_AUDIO_PAD_TOKEN_ID,
    AUDIO_END.id,
]


def _infer_audio_placeholder_token_ids(
    prompt: str,
    *,
    num_audios: int,
    audio_input_token_id: int,
    audio_output_token_id: int,
) -> torch.Tensor:
    """Detect per-audio placeholder type from the prompt text."""
    token_ids: list[int] = []
    for match in AUDIO_PLACEHOLDER_PATTERN.finditer(prompt):
        text = match.group(0)
        if text == AUDIO_PLACEHOLDER_SEQ:
            token_ids.append(audio_input_token_id)
        else:
            token_ids.append(audio_output_token_id)

    if len(token_ids) != num_audios:
        logger.warning(
            "Expected %d audio placeholders, found %d; defaulting to input",
            num_audios, len(token_ids),
        )
        token_ids = [audio_input_token_id] * num_audios

    return torch.tensor(token_ids, dtype=torch.long)


def compute_samples_per_frame(*, sampling_rate: int, frame_rate: float) -> int:
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}.")
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}.")
    ratio = float(sampling_rate) / float(frame_rate)
    rounded = int(round(ratio))
    if abs(ratio - rounded) > 1e-6:
        raise ValueError(f"sampling_rate / frame_rate must be an integer (got {sampling_rate}/{frame_rate}={ratio}).")
    return rounded


def compute_num_audio_input_tokens(
    audio_len_samples: int,
    *,
    sampling_rate: int,
    frame_rate: float,
) -> int:
    if audio_len_samples < 0:
        raise ValueError(f"audio_len_samples must be >= 0, got {audio_len_samples}.")

    samples_per_frame = compute_samples_per_frame(
        sampling_rate=sampling_rate,
        frame_rate=frame_rate,
    )
    return int(math.ceil(audio_len_samples / samples_per_frame))


def normalize_audio_waveforms_and_lengths(
    audio_waveforms: torch.Tensor | list[Any],
    audio_lengths: torch.Tensor | list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch and pad waveforms to ``[N, T]`` float32 and align ``audio_lengths`` ``[N]``.

    Shared by multimodal ``embed_multimodal`` and ``RaonMultiModalProcessor`` inputs.
    """
    if isinstance(audio_waveforms, list):
        waveforms = [torch.as_tensor(w, dtype=torch.float32).reshape(-1) for w in audio_waveforms]
        max_len = max((w.shape[0] for w in waveforms), default=0)
        waveforms = [F.pad(w, (0, max_len - w.shape[0])) if w.shape[0] < max_len else w for w in waveforms]
        stacked = torch.stack(waveforms, dim=0) if waveforms else torch.empty((0, 0), dtype=torch.float32)
    elif isinstance(audio_waveforms, torch.Tensor):
        if audio_waveforms.ndim == 1:
            stacked = audio_waveforms.unsqueeze(0)
        elif audio_waveforms.ndim == 2:
            stacked = audio_waveforms
        else:
            raise ValueError(
                f"audio_waveforms must be 1D/2D tensor or list of 1D tensors, got shape={tuple(audio_waveforms.shape)}."
            )
        stacked = stacked.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported audio_waveforms type: {type(audio_waveforms)}")

    num_audios = int(stacked.shape[0])
    if audio_lengths is None:
        lengths = torch.full((num_audios,), int(stacked.shape[1]), dtype=torch.long)
    elif isinstance(audio_lengths, list):
        lengths = torch.as_tensor(audio_lengths, dtype=torch.long)
    elif isinstance(audio_lengths, torch.Tensor):
        lengths = audio_lengths.to(dtype=torch.long).reshape(-1)
    else:
        raise ValueError(f"Unsupported audio_lengths type: {type(audio_lengths)}")

    if int(lengths.shape[0]) != num_audios:
        raise ValueError(f"audio_lengths size mismatch: got {int(lengths.shape[0])}, expected {num_audios}.")

    return stacked, lengths


def strip_raon_audio_markers(text: str) -> str:
    """Remove audio placeholder / pad tokens from decoded text."""
    if not text:
        return ""
    for token in (
        AUDIO_START_TOKEN,
        AUDIO_END_TOKEN,
        AUDIO_INPUT_PAD_TOKEN,
        AUDIO_OUTPUT_PAD_TOKEN,
        "<|secondary_audio_pad|>",
        "<|audio_pad|>",
    ):
        text = text.replace(token, "")
    return " ".join(text.split())


def flatten_audio_embeddings(
    multimodal_embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(multimodal_embeddings, torch.Tensor):
        if multimodal_embeddings.ndim == 2:
            flat = multimodal_embeddings
        elif multimodal_embeddings.ndim == 3:
            flat = multimodal_embeddings.reshape(-1, multimodal_embeddings.shape[-1])
        else:
            raise ValueError(
                f"multimodal_embeddings tensor must be 2D or 3D, got shape={tuple(multimodal_embeddings.shape)}."
            )
    else:
        if len(multimodal_embeddings) == 0:
            return torch.empty((0, hidden_size), device=device, dtype=dtype)
        flat = torch.cat(multimodal_embeddings, dim=0)

    if flat.shape[-1] != hidden_size:
        raise ValueError(f"audio embedding hidden size mismatch: expected {hidden_size}, got {flat.shape[-1]}.")

    return flat.to(device=device, dtype=dtype)


def scatter_audio_input_embeddings(
    *,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_input_embeddings: torch.Tensor,
    audio_input_token_id: int,
    is_multimodal: torch.Tensor | None,
) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape={tuple(input_ids.shape)}.")
    if inputs_embeds.ndim != 2:
        raise ValueError(f"inputs_embeds must be 2D, got shape={tuple(inputs_embeds.shape)}.")

    audio_mask = input_ids == audio_input_token_id
    if is_multimodal is not None:
        non_audio_mm = is_multimodal & (~audio_mask)
        if bool(non_audio_mm.any().item()):
            raise ValueError(
                "is_multimodal marks tokens that are not audio_input_token_id. "
                "This model only supports audio multimodal placeholders."
            )
        audio_mask = audio_mask & is_multimodal

    num_placeholders = int(audio_mask.sum().item())
    num_embeddings = int(audio_input_embeddings.shape[0])
    if num_placeholders != num_embeddings:
        raise ValueError(
            "Audio embedding alignment error: "
            f"found {num_placeholders} audio_input_token_id placeholders but "
            f"got {num_embeddings} audio embedding frames."
        )

    if num_placeholders == 0:
        return inputs_embeds

    if audio_input_embeddings.ndim != 2:
        raise ValueError(
            f"audio_input_embeddings must be 2D after flattening, got shape={tuple(audio_input_embeddings.shape)}."
        )

    expanded_mask = audio_mask[:, None].expand_as(inputs_embeds)
    if int(expanded_mask.sum().item()) != int(audio_input_embeddings.numel()):
        raise ValueError(
            "Audio embedding scatter shape mismatch: "
            f"mask_elements={int(expanded_mask.sum().item())}, "
            f"embedding_elements={int(audio_input_embeddings.numel())}."
        )

    return inputs_embeds.masked_scatter(expanded_mask, audio_input_embeddings)


def _raon_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    return {
        "audio_waveforms": MultiModalFieldConfig.batched("audio"),
        "audio_lengths": MultiModalFieldConfig.batched("audio"),
        "audio_placeholder_token_ids": MultiModalFieldConfig.batched("audio"),
    }


class RaonMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_waveforms", "audio_lengths"},
                fields_factory=_raon_field_config,
            )

        return super()._parse_audio_data(data)


class RaonProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        hf_config = self.get_hf_config()
        sampling_rate = int(hf_config.audio_tokenizer_config.sampling_rate)
        return RaonMultiModalDataParser(
            target_sr=float(sampling_rate),
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class RaonDummyInputsBuilder(BaseDummyInputsBuilder[RaonProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return AUDIO_PLACEHOLDER_SEQ * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        hf_config = self.info.get_hf_config()
        sampling_rate = int(hf_config.audio_tokenizer_config.sampling_rate)
        target_audio_length = sampling_rate
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


class RaonMultiModalProcessor(BaseMultiModalProcessor[RaonProcessingInfo]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cache = None

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        num_audios = mm_items.get_count("audio", strict=False)
        if num_audios == 0:
            return BatchFeature({}, tensor_type="pt")

        audios = mm_items.get_items("audio", AudioProcessorItems)
        waveform_tensors: list[torch.Tensor] = []
        lengths: list[int] = []
        for i in range(num_audios):
            audio = audios.get(i)
            tensor = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
            waveform_tensors.append(tensor)
            lengths.append(int(tensor.shape[0]))

        audio_waveforms, audio_lengths = normalize_audio_waveforms_and_lengths(waveform_tensors, lengths)

        return BatchFeature(
            {
                "audio_waveforms": audio_waveforms,
                "audio_lengths": audio_lengths,
            },
            tensor_type="pt",
        )

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        del enable_hf_prompt_update

        num_audios = mm_items.get_count("audio", strict=False)
        if isinstance(prompt, str):
            tokenizer = self.info.get_tokenizer()
            align_tokenizer(tokenizer)

            prompt = inject_placeholders_into_str(prompt, num_audios=num_audios)
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)
            if num_audios > 0:
                tokenizer = self.info.get_tokenizer()
                align_tokenizer(tokenizer)
                ph_ids = tokenizer.encode(AUDIO_PLACEHOLDER_SEQ, add_special_tokens=False)
                legacy_ph_ids = tokenizer.encode(
                    f"{AUDIO_START_TOKEN}<|audio_pad|>{AUDIO_END_TOKEN}",
                    add_special_tokens=False,
                )
                marker_ids = tokenizer.encode(f"{USER_PROMPT_MARKER}", add_special_tokens=False)
                prompt_ids = inject_placeholders_into_token_ids(
                    prompt_ids,
                    num_audios=num_audios,
                    ph_ids=ph_ids,
                    legacy_ph_ids=legacy_ph_ids,
                    marker_ids=marker_ids,
                )

        mm_processed_data = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )
        if num_audios > 0:
            tokenizer = self.info.get_tokenizer() if not isinstance(prompt, str) else tokenizer
            align_tokenizer(tokenizer)
            audio_input_tid = int(tokenizer.encode(AUDIO_INPUT_PLACEHOLDER.text, add_special_tokens=False)[0])
            audio_output_tid = int(tokenizer.encode(AUDIO_OUTPUT_PLACEHOLDER.text, add_special_tokens=False)[0])
            prompt_text = prompt if isinstance(prompt, str) else tokenizer.decode(prompt_ids)
            mm_processed_data["audio_placeholder_token_ids"] = _infer_audio_placeholder_token_ids(
                prompt_text,
                num_audios=num_audios,
                audio_input_token_id=audio_input_tid,
                audio_output_token_id=audio_output_tid,
            )
        return prompt_ids, mm_processed_data, False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _raon_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_audios = mm_items.get_count("audio", strict=False)
        if num_audios == 0:
            return []

        hf_config = self.info.get_hf_config()
        atc = hf_config.audio_tokenizer_config
        sampling_rate = int(atc.sampling_rate)
        frame_rate = get_mimi_frame_rate(atc)

        audio_start_token_id = AUDIO_START.id
        audio_end_token_id = AUDIO_END.id
        audio_input_token_id = AUDIO_INPUT_PLACEHOLDER.id
        audio_output_token_id = AUDIO_OUTPUT_PLACEHOLDER.id

        audios = mm_items.get_items("audio", AudioProcessorItems)

        def _build_replacement(
            item_idx: int, *, placeholder_token_id: int, close_with_end: bool,
        ) -> PromptUpdateDetails[list[int]]:
            if hasattr(audios, "get_audio_length"):
                audio_len_samples = int(audios.get_audio_length(item_idx))
            else:
                audio = audios.get(item_idx)
                audio_len_samples = int(len(audio))
            num_audio_tokens = compute_num_audio_input_tokens(
                audio_len_samples,
                sampling_rate=sampling_rate,
                frame_rate=frame_rate,
            )
            if num_audio_tokens == 0:
                raise ValueError(
                    f"The audio input is too short to be represented by Raon: audio_len_samples={audio_len_samples}."
                )
            replacement_ids = [
                audio_start_token_id,
                *([placeholder_token_id] * num_audio_tokens),
            ]
            if close_with_end:
                replacement_ids.append(audio_end_token_id)
            return PromptUpdateDetails.select_token_id(
                replacement_ids,
                embed_token_id=placeholder_token_id,
            )

        def get_input_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            return _build_replacement(item_idx, placeholder_token_id=audio_input_token_id, close_with_end=True)

        def get_output_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            return _build_replacement(item_idx, placeholder_token_id=audio_output_token_id, close_with_end=True)

        def get_output_open_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            return _build_replacement(item_idx, placeholder_token_id=audio_output_token_id, close_with_end=False)

        return [
            PromptReplacement(
                modality="audio",
                target=AUDIO_PLACEHOLDER_SEQ,
                replacement=get_input_replacement,
            ),
            PromptReplacement(
                modality="audio",
                target=LEGACY_AUDIO_PLACEHOLDER_TOKEN_IDS,
                replacement=get_input_replacement,
            ),
            PromptReplacement(
                modality="audio",
                target=AUDIO_OUTPUT_PLACEHOLDER_SEQ,
                replacement=get_output_replacement,
            ),
            PromptReplacement(
                modality="audio",
                target=AUDIO_OUTPUT_OPEN_SEQ,
                replacement=get_output_open_replacement,
            ),
        ]
