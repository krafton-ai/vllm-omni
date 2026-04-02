# SPDX-License-Identifier: Apache-2.0
"""Speaker encoder modules for Raon TTS."""

from __future__ import annotations

import base64
import dataclasses
import io
import os
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi
import torchaudio.functional
from torch import nn

try:
    import soundfile
except ImportError:
    soundfile = None

from vllm_omni.transformers_utils.configs.raon import (
    AUDIO_SAMPLE_RATE,
    TARGET_ENCODER_SAMPLE_RATE,
    SpeakerEncoderConfig,
)
from vllm.logger import init_logger
from vllm_omni.model_executor.models.raon.raon_utils import (
    module_device,
    module_dtype,
    unwrap_singleton_list,
)

logger = init_logger(__name__)


def _coerce_frame_lengths(
    lengths: torch.Tensor | None,
    *,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    if lengths is None:
        return torch.full((1,), max_len, device=device, dtype=torch.long)
    lengths = torch.as_tensor(lengths, device=device)
    if lengths.ndim == 0:
        lengths = lengths.unsqueeze(0)
    if torch.is_floating_point(lengths):
        if float(lengths.max().item()) <= 1.0 + 1e-6:
            lengths = lengths * max_len
        lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.long).clamp(min=1, max=max_len)


def _length_to_mask(
    lengths: torch.Tensor | None,
    *,
    max_len: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    resolved_device = device or (lengths.device if isinstance(lengths, torch.Tensor) else torch.device("cpu"))
    frame_lengths = _coerce_frame_lengths(lengths, max_len=max_len, device=resolved_device)
    mask = torch.arange(max_len, device=resolved_device).expand(len(frame_lengths), max_len) < frame_lengths.unsqueeze(1)
    if dtype is None:
        return mask
    return mask.to(dtype=dtype)


@dataclasses.dataclass
class _EcapaConfig:
    mel_dim: int = 80
    enc_dim: int = 192
    enc_channels: list[int] = dataclasses.field(default_factory=lambda: [1024, 1024, 1024, 1024, 3072])
    enc_kernel_sizes: list[int] = dataclasses.field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    enc_attention_channels: int = 128


class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()
        self.norm = BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(p=0.0)

    def forward(self, hidden_states: torch.Tensor):
        return self.dropout(self.norm(self.activation(self.conv(hidden_states))))


class BatchNorm1d(nn.Module):
    """Wrapper that mirrors SpeechBrain's ``BatchNorm1d`` state-dict layout."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation)
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)
        else:
            mask = _length_to_mask(
                lengths,
                max_len=hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ).unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
            hidden_states_mean = (hidden_states * mask).sum(dim=2, keepdim=True) / total
        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))
        return hidden_states * hidden_states_mean


class SqueezeExcitationRes2NetBlock(nn.Module):
    """TDNN-Res2Net-TDNN-SE building block used in ECAPA-TDNN."""

    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state, lengths=lengths)
        return hidden_state + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistic pooling layer: returns concatenated mean and std."""

    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1, padding="same", padding_mode="reflect")

    @staticmethod
    def _compute_statistics(x, m, dim=2, eps=1e-12):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
        return mean, std

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        seq_length = hidden_states.shape[-1]
        if lengths is None:
            lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)
        mask = _length_to_mask(
            lengths,
            max_len=seq_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        mask = mask.unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)
        attention = self.conv(self.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        pooled_stats = torch.cat((mean, std), dim=1)
        return pooled_stats.unsqueeze(2)


class EcapaTdnnEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder.

    Reference: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://huggingface.co/papers/2005.07143).
    """

    def __init__(self, config: _EcapaConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1], config.enc_channels[-1], config.enc_kernel_sizes[-1], config.enc_dilations[-1]
        )
        self.asp = AttentiveStatisticsPooling(config.enc_channels[-1], attention_channels=config.enc_attention_channels)
        self.asp_bn = BatchNorm1d(config.enc_channels[-1] * 2)
        self.fc = nn.Conv1d(
            config.enc_channels[-1] * 2,
            config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states_list = []
        for idx, layer in enumerate(self.blocks):
            if idx == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(hidden_states, lengths=lengths)
            hidden_states_list.append(hidden_states)
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)
        hidden_states = self.asp(hidden_states, lengths=lengths)
        hidden_states = self.asp_bn(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states.squeeze(-1)


_ECAPA_CONFIG = _EcapaConfig()


# Pretrained weights loaded from the SpeechBrain ECAPA-TDNN checkpoint
# (Apache-2.0, https://github.com/speechbrain/speechbrain).
# The encoder architecture above is a clean-room reimplementation from:
# Desplanques et al., "ECAPA-TDNN" (https://arxiv.org/abs/2005.07143).


def _map_speechbrain_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map speechbrain ``embedding_model.ckpt`` keys to ``EcapaTdnnEncoder`` keys.

    The main difference is that speechbrain wraps Conv1d in an extra container,
    producing keys like ``blocks.0.conv.conv.weight`` where our module expects
    ``blocks.0.conv.weight``.  We collapse the extra wrapper level in three
    patterns:

    1. ``.conv.conv.`` → ``.conv.`` (double-wrapped Conv1d)
    2. ``fc.conv.`` → ``fc.`` (final fully-connected conv)
    3. ``.convN.conv.`` → ``.convN.`` (single-wrapped, e.g. SE block conv1/conv2)

    """
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(".conv.conv.", ".conv.")
        new_key = new_key.replace("fc.conv.", "fc.")
        # SE block: se_block.conv1.conv.{w,b} → se_block.conv1.{w,b}
        new_key = re.sub(r"(\.conv\d+)\.conv\.", r"\1.", new_key)

        mapped[new_key] = value
    return mapped


def _load_ecapa_from_hf(model_id: str) -> EcapaTdnnEncoder:
    from huggingface_hub import hf_hub_download

    model = EcapaTdnnEncoder(_ECAPA_CONFIG)

    ckpt_path = hf_hub_download(repo_id=model_id, filename="embedding_model.ckpt")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mapped = _map_speechbrain_keys(state_dict)

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        logger.warning("ECAPA-TDNN load: missing keys: %s", missing)
    if unexpected:
        logger.warning("ECAPA-TDNN load: unexpected keys: %s", unexpected)

    model.eval()
    return model


class EcapaSpeakerBackend(nn.Module):
    """Drop-in replacement for the speechbrain ``EncoderClassifier`` backend.

    Provides the same ``encode_batch(audio_16k, wav_lens)`` interface used by
    ``PretrainedSpeakerEncoder._extract_embedding``.
    """

    def __init__(self, ecapa: EcapaTdnnEncoder) -> None:
        super().__init__()
        self.ecapa = ecapa

    @staticmethod
    def _compute_fbank(
        waveform: torch.Tensor,
        wav_lens: torch.Tensor | None = None,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 80-bin log-mel features close to SpeechBrain's default path."""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            f_max=float(sample_rate // 2),
            n_mels=80,
            window_fn=torch.hamming_window,
            power=2.0,
            center=True,
            pad_mode="constant",
            normalized=False,
        ).to(device=waveform.device)
        sample_lengths = _coerce_frame_lengths(wav_lens, max_len=int(waveform.shape[1]), device=waveform.device)
        feats_list = []
        feat_lengths: list[int] = []
        for i in range(waveform.shape[0]):
            sample_len = int(sample_lengths[min(i, len(sample_lengths) - 1)].item())
            mel = mel_transform(waveform[i : i + 1, :sample_len].float())
            feat = torch.clamp(mel.squeeze(0).transpose(0, 1), min=1e-10)
            feat = 10.0 * torch.log10(feat)
            feat = torch.maximum(feat, feat.max() - 80.0)
            feat = feat.to(dtype=waveform.dtype)
            feats_list.append(feat)
            feat_lengths.append(int(feat.shape[0]))

        max_len = max(f.shape[0] for f in feats_list)
        batch = torch.zeros(len(feats_list), max_len, 80, device=waveform.device, dtype=waveform.dtype)
        for i, feat in enumerate(feats_list):
            batch[i, : feat.shape[0]] = feat
        return batch, torch.tensor(feat_lengths, device=waveform.device, dtype=torch.long)

    @staticmethod
    def _normalize(feats: torch.Tensor, feat_lengths: torch.Tensor) -> torch.Tensor:
        """Per-utterance mean-only normalization (matches speechbrain std_norm=False)."""
        mask = _length_to_mask(
            feat_lengths,
            max_len=int(feats.shape[1]),
            dtype=feats.dtype,
            device=feats.device,
        ).unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (feats * mask).sum(dim=1, keepdim=True) / denom
        return feats - mean

    def encode_batch(
        self, audio_16k: torch.Tensor, wav_lens: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Produce speaker embeddings matching speechbrain's output shape ``[B, 1, 192]``."""
        feats, feat_lengths = self._compute_fbank(audio_16k, wav_lens=wav_lens)
        feats = self._normalize(feats, feat_lengths)
        feats = feats.to(device=next(self.ecapa.parameters()).device)
        feat_lens = feat_lengths.float() / feat_lengths.max().float().clamp(min=1.0)
        embeddings = self.ecapa(feats, lengths=feat_lens.to(device=feats.device))
        return embeddings.unsqueeze(1)


def _soundfile_to_tensor(source: str | io.BytesIO) -> tuple[torch.Tensor, int]:
    if soundfile is None:
        raise RuntimeError("soundfile is unavailable.")
    audio_np, sr = soundfile.read(source, dtype="float32", always_2d=True)
    audio = torch.from_numpy(np.asarray(audio_np, dtype=np.float32)).transpose(0, 1).contiguous()
    return audio, int(sr)


class PretrainedSpeakerEncoder(nn.Module):
    """Speaker encoder using ECAPA-TDNN with a frozen backend and trainable projection."""

    def __init__(self, config: SpeakerEncoderConfig, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if config.pretrained_dim is None:
            raise ValueError(f"pretrained_dim must be set for encoder_type={config.encoder_type}")

        self.encoder_type = config.encoder_type
        self.pretrained_model_id = config.pretrained_model_id
        self.pretrained_dim = int(config.pretrained_dim)
        self.output_size = int(config.output_size)
        self.source_sample_rate = AUDIO_SAMPLE_RATE
        self.target_sample_rate = TARGET_ENCODER_SAMPLE_RATE

        self.projection = nn.Linear(self.pretrained_dim, self.output_size, bias=False, dtype=dtype)

        self._backend: Any = None
        self._backend_device = torch.device("cpu")

    def _load_backend(self) -> None:
        if self._backend is not None:
            return

        model_id = self.pretrained_model_id or "speechbrain/spkrec-ecapa-voxceleb"
        ecapa = _load_ecapa_from_hf(model_id)
        backend = EcapaSpeakerBackend(ecapa)

        object.__setattr__(self, "_backend", backend)
        for param in self._backend.parameters():
            param.requires_grad = False
        self._backend_device = torch.device("cpu")

    def warm_backend_artifacts(self) -> None:
        """Populate local caches for the pretrained backend without binding it."""
        model_id = self.pretrained_model_id or "speechbrain/spkrec-ecapa-voxceleb"
        ecapa = _load_ecapa_from_hf(model_id)
        del ecapa

    def _ensure_backend_device(self) -> None:
        if self._backend is None:
            self._load_backend()

        target_device = self.projection.weight.device
        if self._backend_device == target_device:
            return

        self._backend.to(target_device)
        self._backend_device = target_device
        logger.info("Speaker backend ready on %s", target_device)

    def _extract_embedding(self, audio_16k: torch.Tensor, lengths_16k: torch.Tensor) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError("Pretrained speaker backend is not initialized.")

        min_samples_16k = 1600
        batch = int(audio_16k.shape[0])
        if int(audio_16k.shape[1]) < min_samples_16k:
            return torch.zeros(batch, self.pretrained_dim, device=audio_16k.device, dtype=audio_16k.dtype)
        lengths_16k = lengths_16k.clamp(min=min_samples_16k)
        wav_lens = lengths_16k.float() / lengths_16k.max().float()
        with torch.no_grad():
            embeddings = self._backend.encode_batch(audio_16k, wav_lens)
        return embeddings.squeeze(1)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_backend_device()
        if self._backend is None:
            raise RuntimeError("Pretrained speaker backend is unavailable.")

        if audio.ndim == 3:
            if int(audio.shape[1]) != 1:
                raise ValueError(
                    "Pretrained speaker encoder expects mono audio when 3D input is provided, "
                    f"got shape={tuple(audio.shape)}."
                )
            audio = audio[:, 0]
        elif audio.ndim != 2:
            raise ValueError(f"Pretrained speaker encoder expects audio as [B, T], got shape={tuple(audio.shape)}.")

        audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long).reshape(-1)
        if int(audio_lengths.shape[0]) != int(audio.shape[0]):
            raise ValueError(
                "audio_lengths batch mismatch for speaker encoder: "
                f"got {int(audio_lengths.shape[0])}, expected {int(audio.shape[0])}."
            )

        audio_16k = torchaudio.functional.resample(
            audio.float(),
            orig_freq=self.source_sample_rate,
            new_freq=self.target_sample_rate,
        )
        lengths_16k = (audio_lengths.float() * self.target_sample_rate / self.source_sample_rate).long()
        lengths_16k = lengths_16k.clamp(min=1, max=int(audio_16k.shape[1]))

        raw_embedding = self._extract_embedding(audio_16k, lengths_16k)
        raw_embedding = raw_embedding.to(dtype=self.projection.weight.dtype)
        projected = self.projection(raw_embedding)
        logger.debug("Speaker projected: in=%s out=%s", tuple(audio.shape), tuple(projected.shape))
        return projected.unsqueeze(1)


def build_speaker_encoder(
    config: SpeakerEncoderConfig,
    dtype: torch.dtype | None = None,
) -> PretrainedSpeakerEncoder:
    return PretrainedSpeakerEncoder(config, dtype=dtype)


def normalize_speaker_ref_audio(value: Any) -> str | None:
    value = unwrap_singleton_list(value)

    if isinstance(value, dict):
        for key in ("speaker_ref_audio", "ref_audio", "audio_path", "path", "value"):
            if key in value:
                return normalize_speaker_ref_audio(value.get(key))
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    if hasattr(value, "__fspath__"):
        value = os.fspath(value)  # type: ignore[arg-type]

    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized if normalized else None


def load_speaker_ref_audio(path: str) -> tuple[torch.Tensor, int] | None:
    normalized_path = path.strip()
    if not normalized_path:
        return None

    if normalized_path.startswith("data:"):
        try:
            _, payload = normalized_path.split(",", 1)
            audio_bytes = base64.b64decode(payload)
            buffer = io.BytesIO(audio_bytes)
            if soundfile is not None:
                audio, sr = _soundfile_to_tensor(buffer)
            else:
                audio, sr = torchaudio.load(buffer)
                sr = int(sr)
            logger.info("Loaded ref from data URL: shape=%s sr=%s", tuple(audio.shape), sr)
            return audio, sr
        except Exception as exc:
            logger.warning("Failed to decode data URL ref audio: %s", exc)
            return None

    if not os.path.isfile(normalized_path):
        logger.warning("Ref audio path does not exist: %s", normalized_path[:200])
        return None

    if normalized_path.lower().endswith(".wav") and soundfile is not None:
        try:
            audio, sr = _soundfile_to_tensor(normalized_path)
            logger.info(
                "Loaded WAV: path=%s shape=%s sr=%s",
                normalized_path[:200],
                tuple(audio.shape),
                sr,
            )
            return audio, sr
        except Exception as exc:
            logger.warning(
                "Soundfile WAV load failed for %s; falling back to torchaudio: %s",
                normalized_path,
                exc,
            )

    try:
        audio, sr = torchaudio.load(normalized_path)
        sr = int(sr)
        logger.info(
            "Loaded ref audio: path=%s shape=%s sr=%s",
            normalized_path[:200],
            tuple(audio.shape),
            sr,
        )
        return audio, sr
    except Exception as torchaudio_exc:
        if soundfile is None:
            logger.warning(
                "Torchaudio load failed for %s and soundfile is unavailable: %s",
                normalized_path,
                torchaudio_exc,
            )
            return None

        try:
            audio, sr = _soundfile_to_tensor(normalized_path)
            return audio, sr
        except Exception as soundfile_exc:
            logger.warning(
                "Failed to load ref audio %s (torchaudio error=%s, soundfile error=%s).",
                normalized_path,
                torchaudio_exc,
                soundfile_exc,
            )
            return None


@torch.inference_mode()
def compute_speaker_embeds(
    speaker_encoder: PretrainedSpeakerEncoder,
    *,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor | None = None,
    sampling_rate: int | None = None,
    model_sampling_rate: int,
) -> torch.Tensor:
    if speaker_encoder is None:
        raise AssertionError("speaker_encoder is not initialized on this checkpoint.")

    if audio.ndim == 1:
        audio = audio[None, None]
    elif audio.ndim == 2:
        audio = audio[:, None]
    elif audio.ndim != 3:
        raise ValueError(f"speaker audio must be 1D/2D/3D tensor, got shape={tuple(audio.shape)}.")

    target_device = module_device(speaker_encoder)
    target_dtype = module_dtype(speaker_encoder)
    audio = audio.to(device=target_device, dtype=target_dtype)

    if audio_lengths is None:
        audio_lengths = torch.full(
            (int(audio.shape[0]),),
            int(audio.shape[-1]),
            dtype=torch.long,
            device=audio.device,
        )
    else:
        audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long).reshape(-1)
        if int(audio_lengths.shape[0]) != int(audio.shape[0]):
            raise ValueError(
                "audio_lengths batch mismatch for speaker embeddings: "
                f"got {int(audio_lengths.shape[0])}, expected {int(audio.shape[0])}."
            )

    if sampling_rate is not None and sampling_rate != model_sampling_rate:
        audio = torchaudio.functional.resample(
            audio,
            orig_freq=int(sampling_rate),
            new_freq=int(model_sampling_rate),
        )
        audio_lengths = (audio_lengths.float() * float(model_sampling_rate) / float(sampling_rate)).long()

    audio_lengths = audio_lengths.clamp(min=1, max=int(audio.shape[-1]))

    embeds = speaker_encoder(audio[:, 0], audio_lengths)
    logger.debug("Pretrained embeds: shape=%s sr=%s", tuple(embeds.shape), sampling_rate)
    return embeds
