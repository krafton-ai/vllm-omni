# SPDX-License-Identifier: Apache-2.0
"""Audio encoder wrappers for Raon (Qwen3Omni AuT and Mimi-based)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from transformers import WhisperFeatureExtractor
from transformers.cache_utils import Cache
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from transformers.utils.generic import ModelOutput

from vllm_omni.transformers_utils.configs.raon import (
    AUDIO_SAMPLE_RATE,
    TARGET_ENCODER_SAMPLE_RATE,
)
from vllm_omni.model_executor.models.raon.raon_multimodal import compute_samples_per_frame


@dataclass
class RaonAudioEncoderOutput(ModelOutput):
    embeds: torch.Tensor | None = None
    encoder_past_key_values: Cache | None = None
    padding_cache: Any = None


class Qwen3OmniAuTWrapper(nn.Module):
    config: Qwen3OmniMoeAudioEncoderConfig

    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        feature_extractor: WhisperFeatureExtractor,
        encoder: Qwen3OmniMoeAudioEncoder,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = feature_extractor
        self.encoder = encoder

        self.input_sample_rate = AUDIO_SAMPLE_RATE
        self.encoder_sample_rate = TARGET_ENCODER_SAMPLE_RATE
        self.frame_rate = 12.5
        self.hidden_size = int(config.output_dim)
        self._min_encoder_samples = int(feature_extractor.n_fft)

        self.config.sampling_rate = self.input_sample_rate

    @classmethod
    def from_config(
        cls,
        config: Qwen3OmniMoeAudioEncoderConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Qwen3OmniAuTWrapper:
        feature_extractor = WhisperFeatureExtractor(
            feature_size=int(getattr(config, "num_mel_bins", 128)),
            sampling_rate=TARGET_ENCODER_SAMPLE_RATE,
        )
        encoder = Qwen3OmniMoeAudioEncoder(config)
        encoder = encoder.to(dtype=dtype)
        return cls(config=config, feature_extractor=feature_extractor, encoder=encoder)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def compute_expected_output_length(self, num_samples: int) -> int:
        samples_per_frame = compute_samples_per_frame(
            sampling_rate=self.input_sample_rate,
            frame_rate=self.frame_rate,
        )
        return math.ceil(num_samples / samples_per_frame)

    @staticmethod
    def _compute_encoder_output_lengths(feature_lens: list[int]) -> list[int]:
        output_lens: list[int] = []
        for feature_len in feature_lens:
            input_lengths_leave = feature_len % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            output_len = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_len // 100) * 13
            output_lens.append(int(output_len))
        return output_lens

    @staticmethod
    def _match_expected_output_length(
        embeds: torch.Tensor,
        *,
        expected_output_length: int,
    ) -> torch.Tensor:
        actual_output_length = int(embeds.shape[1])
        if actual_output_length > expected_output_length:
            return embeds[:, :expected_output_length]
        if actual_output_length < expected_output_length:
            return F.interpolate(
                embeds.transpose(1, 2),
                size=expected_output_length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return embeds

    def forward(
        self,
        audio: torch.Tensor,
        encoder_past_key_values: Cache | None = None,
        padding_cache: Any = None,
        use_streaming: bool | None = None,
    ) -> RaonAudioEncoderOutput:
        if encoder_past_key_values is not None:
            raise ValueError("Qwen3OmniAuTWrapper does not support encoder_past_key_values.")
        if padding_cache is not None:
            raise ValueError("Qwen3OmniAuTWrapper does not support padding_cache.")
        if use_streaming:
            raise ValueError("Qwen3OmniAuTWrapper does not support streaming mode.")

        if not (1 <= audio.shape[1] <= 2):
            raise ValueError(f"Number of audio channels must be 1 or 2, got {audio.shape[1]}.")

        num_samples = int(audio.shape[2])
        expected_output_length = self.compute_expected_output_length(num_samples)

        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)
        audio = audio.squeeze(1)

        if self.input_sample_rate != self.encoder_sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            )

        # Whisper STFT requires at least n_fft samples.
        if audio.shape[-1] < self._min_encoder_samples:
            audio = F.pad(audio, (0, self._min_encoder_samples - audio.shape[-1]))

        audio_features = self.feature_extractor(
            raw_speech=[s.float().cpu().numpy() for s in audio],
            sampling_rate=self.encoder_sample_rate,
            return_tensors="pt",
            padding=True,  # type: ignore[arg-type]
            return_attention_mask=True,
        )
        input_features = audio_features["input_features"]
        feature_attention_mask = audio_features["attention_mask"]

        # Pack only non-padded frames and pass aligned lengths to the encoder.
        feature_lens = feature_attention_mask.sum(dim=1).to(device=self.device, dtype=torch.long)
        packed_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        packed_features = packed_features.to(device=self.device, dtype=self.dtype)

        outputs = self.encoder(input_features=packed_features, feature_lens=feature_lens)
        last_hidden_state = outputs.last_hidden_state
        if not isinstance(last_hidden_state, torch.Tensor):
            raise RuntimeError("Qwen3OmniAuTWrapper encoder returned no last_hidden_state.")

        output_lens = self._compute_encoder_output_lengths(feature_lens.tolist())

        hidden_states_list = list(torch.split(last_hidden_state, output_lens, dim=0))
        embeds = nn.utils.rnn.pad_sequence(hidden_states_list, batch_first=True)
        embeds = self._match_expected_output_length(
            embeds,
            expected_output_length=expected_output_length,
        )

        return RaonAudioEncoderOutput(
            embeds=embeds,
            encoder_past_key_values=None,
            padding_cache=None,
        )


__all__ = ["RaonAudioEncoderOutput", "Qwen3OmniAuTWrapper"]
