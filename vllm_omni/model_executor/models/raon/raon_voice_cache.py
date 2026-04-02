# SPDX-License-Identifier: Apache-2.0
"""Raon TTS voice-clone cache (safetensors only; no pickle)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.metadata_manager import MetadataManager

logger = init_logger(__name__)


@dataclass
class RaonVoiceClonePromptItem:
    """One voice-clone prompt (embedding + mode flags + optional ref text)."""

    ref_spk_embedding: torch.Tensor  # (D,) ECAPA-TDNN speaker embedding
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None
    ref_codec_codes: torch.Tensor | None = None       # [T, G] raw Mimi RVQ codes, dtype int64
    ref_codec_codes_mask: torch.Tensor | None = None   # [T] validity mask, dtype bool


class RaonVoiceCacheManager:
    """Speaker embedding cache; safetensors under the configured samples directory."""

    def __init__(
        self,
        speech_voice_samples_dir: str | None = None,
        metadata_manager: MetadataManager | None = None,
    ) -> None:
        self.speech_voice_samples_dir = speech_voice_samples_dir or os.environ.get(
            "SPEECH_VOICE_SAMPLES", "/tmp/voice_samples"
        )

        if metadata_manager is not None:
            self.metadata_manager = metadata_manager
        else:
            metadata_file = Path(self.speech_voice_samples_dir) / "metadata.json"
            self.metadata_manager = MetadataManager(metadata_file)

    def load_uploaded_speakers_from_metadata(self) -> dict[str, Any] | None:
        try:
            return self.metadata_manager.get_uploaded_speakers()
        except Exception as e:
            logger.warning("Failed to load uploaded speakers from metadata: %s", e)
            return None

    def update_metadata_cache_info(
        self, speaker: str, cache_file_path: Path, status: str = "ready"
    ) -> bool:
        try:
            return self.metadata_manager.update_cache_info(
                speaker_key=speaker.lower(),
                cache_file_path=cache_file_path,
                status=status,
            )
        except Exception as e:
            logger.error("Failed to update metadata cache info: %s", e)
            return False

    def save_voice_cache(
        self,
        speaker: str,
        audio_file_path: Path,
        prompt_item: RaonVoiceClonePromptItem,
    ) -> bool:
        """Save a single voice clone prompt item as safetensors."""
        try:
            cache_file_path = audio_file_path.with_suffix(".safetensors")

            tensors: dict[str, torch.Tensor] = {
                "ref_spk_embedding": prompt_item.ref_spk_embedding.detach().cpu(),
                "x_vector_only_mode": torch.tensor(
                    int(prompt_item.x_vector_only_mode), dtype=torch.int8
                ),
                "icl_mode": torch.tensor(
                    int(prompt_item.icl_mode), dtype=torch.int8
                ),
            }

            # Mimi codec codes cache
            has_ref_codec_codes = prompt_item.ref_codec_codes is not None
            tensors["has_ref_codec_codes"] = torch.tensor(int(has_ref_codec_codes), dtype=torch.int8)
            if has_ref_codec_codes:
                tensors["ref_codec_codes"] = prompt_item.ref_codec_codes.detach().cpu()
                if prompt_item.ref_codec_codes_mask is not None:
                    tensors["ref_codec_codes_mask"] = prompt_item.ref_codec_codes_mask.detach().cpu()

            metadata: dict[str, str] = {"cache_version": "2"}
            if prompt_item.ref_text is not None:
                metadata["ref_text"] = prompt_item.ref_text

            # Atomic write: temp file in same dir → rename
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".safetensors", dir=str(cache_file_path.parent),
            )
            os.close(tmp_fd)
            try:
                save_file(tensors, tmp_path, metadata=metadata)
                os.rename(tmp_path, str(cache_file_path))
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            return self.update_metadata_cache_info(
                speaker=speaker,
                cache_file_path=cache_file_path,
                status="ready",
            )
        except Exception as e:
            logger.error("Failed to save safetensors cache for speaker %s: %s", speaker, e)
            self.update_metadata_cache_info(speaker, Path(""), "failed")
            return False

    def load_cached_voice_prompt(
        self,
        speaker: str,
        device: str | None = None,
    ) -> RaonVoiceClonePromptItem | None:
        """Load cached voice clone prompt from safetensors."""
        try:
            uploaded_speakers = self.load_uploaded_speakers_from_metadata()
            if not uploaded_speakers:
                return None

            speaker_key = speaker.lower()
            if speaker_key not in uploaded_speakers:
                return None

            speaker_info = uploaded_speakers[speaker_key]
            if speaker_info.get("cache_status") != "ready":
                return None

            cache_file_path = Path(speaker_info.get("cache_file", "")).resolve()
            base_dir = Path(self.speech_voice_samples_dir).resolve()

            # Path confinement: use trailing os.sep to avoid prefix collisions.
            base_prefix = str(base_dir).rstrip("/") + "/"
            if not str(cache_file_path).startswith(base_prefix):
                logger.error("Illegal cache path outside base dir: %s", cache_file_path)
                return None

            if not cache_file_path.exists():
                return None

            if cache_file_path.suffix != ".safetensors":
                logger.error("Legacy or unsafe cache format rejected: %s", cache_file_path)
                return None

            with safe_open(cache_file_path, framework="pt", device="cpu") as f:
                meta = f.metadata()
                ref_spk_embedding = f.get_tensor("ref_spk_embedding")
                if device is not None:
                    ref_spk_embedding = ref_spk_embedding.to(device)

                x_vector_only_mode = bool(f.get_tensor("x_vector_only_mode").item())
                icl_mode = bool(f.get_tensor("icl_mode").item())
                ref_text = meta.get("ref_text") if meta else None

                # Backward-compatible: load Mimi codec codes if present
                ref_codec_codes = None
                ref_codec_codes_mask = None
                if "has_ref_codec_codes" in f.keys():
                    has_codes = bool(f.get_tensor("has_ref_codec_codes").item())
                    if has_codes and "ref_codec_codes" in f.keys():
                        ref_codec_codes = f.get_tensor("ref_codec_codes")
                        if device is not None:
                            ref_codec_codes = ref_codec_codes.to(device)
                        if "ref_codec_codes_mask" in f.keys():
                            ref_codec_codes_mask = f.get_tensor("ref_codec_codes_mask")
                            if device is not None:
                                ref_codec_codes_mask = ref_codec_codes_mask.to(device)

            logger.info("Safetensors cache loaded for speaker: %s", speaker)
            return RaonVoiceClonePromptItem(
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=x_vector_only_mode,
                icl_mode=icl_mode,
                ref_text=ref_text,
                ref_codec_codes=ref_codec_codes,
                ref_codec_codes_mask=ref_codec_codes_mask,
            )
        except Exception as e:
            logger.warning("Failed to load safetensors cache for speaker %s: %s", speaker, e)
            return None

    def get_speaker_audio_path(self, speaker: str) -> Path | None:
        uploaded_speakers = self.load_uploaded_speakers_from_metadata()
        if not uploaded_speakers:
            return None

        speaker_key = speaker.lower()
        if speaker_key not in uploaded_speakers:
            return None

        audio_file_path = Path(uploaded_speakers[speaker_key]["file_path"])
        if audio_file_path.exists():
            return audio_file_path

        logger.warning("Audio file not found for speaker %s: %s", speaker, audio_file_path)
        return None
