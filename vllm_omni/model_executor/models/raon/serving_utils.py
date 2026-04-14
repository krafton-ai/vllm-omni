# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import io
import re
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml as _yaml
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.raon.raon_audio_encoder import (
    compute_num_audio_input_tokens,
)
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END,
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_START_TOKEN,
    IM_END,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    RaonChatTemplateBuilder,
    TaskType,
    normalize_token_ids,
)
from vllm_omni.transformers_utils.configs.raon import (
    AUDIO_SAMPLE_RATE,
    ENV,
    TTS_MAX_TOKENS_HARD_CAP,
)

logger = init_logger(__name__)

_DEFAULT_SPEAKER_NPY = Path(__file__).resolve().parent / "assets" / "default_speaker.npy"

# Lazily-cached default speaker embedding (192-dim ECAPA x-vector).
_default_speaker_embedding: torch.Tensor | None = None


def decode_audio_data_url(data_url: str) -> tuple[np.ndarray, int]:
    """Decode a ``data:audio/...;base64,...`` URL to (waveform, sample_rate).

    Only handles ``data:`` URIs.  Raises :class:`ValueError` for other schemes.
    """
    if not data_url.startswith("data:"):
        raise ValueError(f"Expected data: URI, got {data_url[:40]!r}...")
    b64_data = data_url.split(",", 1)[1] if "," in data_url else data_url
    audio_bytes = base64.b64decode(b64_data)
    audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]  # mono
    return audio_np, int(sr)


def get_default_speaker_embedding() -> torch.Tensor | None:
    """Return the pre-computed 192-dim ECAPA embedding for the default speaker."""
    global _default_speaker_embedding
    if _default_speaker_embedding is not None:
        return _default_speaker_embedding.clone()
    if not _DEFAULT_SPEAKER_NPY.is_file():
        logger.warning("Default speaker embedding not found at %s", _DEFAULT_SPEAKER_NPY)
        return None
    _default_speaker_embedding = torch.from_numpy(np.load(str(_DEFAULT_SPEAKER_NPY))).to(dtype=torch.float32)
    logger.info(
        "Loaded default speaker embedding: shape=%s, norm=%.2f",
        tuple(_default_speaker_embedding.shape),
        _default_speaker_embedding.norm().item(),
    )
    return _default_speaker_embedding.clone()


class _NullTokenizer:
    """Sentinel used when no real tokenizer is available."""


MODALITY_TEXT = "text"
MODALITY_AUDIO = "audio"

OUTPUT_MODE_TEXT_ONLY = "text_only"
OUTPUT_MODE_AUDIO_ONLY = "audio_only"
OUTPUT_MODE_TEXT_AND_AUDIO = "text_and_audio"

_GLOBAL_TOP_K = 20

# ---------------------------------------------------------------------------
# Task-specific sampling params loaded from raon.yaml (lazy-cached).
# ---------------------------------------------------------------------------
_TASK_SAMPLING_PARAMS: dict[str, dict[str, Any]] | None = None


def _get_task_sampling_params() -> dict[str, dict[str, Any]]:
    """Load task-specific sampling defaults from raon.yaml (cached)."""
    global _TASK_SAMPLING_PARAMS
    if _TASK_SAMPLING_PARAMS is not None:
        return _TASK_SAMPLING_PARAMS
    yaml_path = Path(__file__).resolve().parents[2] / "stage_configs" / "raon.yaml"
    try:
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        for stage in cfg.get("stage_args", []):
            if stage.get("is_comprehension"):
                _TASK_SAMPLING_PARAMS = stage.get("task_sampling_params", {})
                logger.info("Loaded Raon task_sampling_params: %s", list(_TASK_SAMPLING_PARAMS.keys()))
                return _TASK_SAMPLING_PARAMS
    except Exception:
        logger.warning("Failed to load task_sampling_params from %s", yaml_path, exc_info=True)
    _TASK_SAMPLING_PARAMS = {}
    return _TASK_SAMPLING_PARAMS


# Shared helpers: modalities, tokenizer invariants, TTS request seeding.


def canonicalize_modalities(
    requested_modalities: Sequence[str] | None,
    default_modalities: Sequence[str] | None,
) -> list[str]:
    source = requested_modalities if requested_modalities is not None else default_modalities
    if source is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for modality in source:
        if not isinstance(modality, str):
            continue
        value = modality.strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def modalities_to_output_mode(modalities: Sequence[str] | None) -> str | None:
    if modalities is None:
        return None

    normalized = set(canonicalize_modalities(modalities, None))
    if normalized == {MODALITY_TEXT}:
        return OUTPUT_MODE_TEXT_ONLY
    if normalized == {MODALITY_AUDIO}:
        return OUTPUT_MODE_AUDIO_ONLY
    if normalized == {MODALITY_TEXT, MODALITY_AUDIO}:
        return OUTPUT_MODE_TEXT_AND_AUDIO
    return None


def attach_output_mode_additional_information(
    prompt: MutableMapping[str, Any],
    output_mode: str | None,
) -> None:
    if output_mode is None:
        return

    additional_information = prompt.get("additional_information")
    if not isinstance(additional_information, dict):
        additional_information = {}
    additional_information["output_mode"] = [output_mode]
    prompt["additional_information"] = additional_information


def extract_audio_data_and_sample_rate(mm_output: MutableMapping[str, Any]) -> tuple[Any, int]:
    audio_data = mm_output.get("audio")
    if audio_data is None:
        audio_data = mm_output.get("model_outputs")

    sample_rate = mm_output.get("sr", AUDIO_SAMPLE_RATE)
    if isinstance(sample_rate, list) and sample_rate:
        sample_rate = sample_rate[-1]
    if isinstance(sample_rate, torch.Tensor):
        if sample_rate.numel() == 1:
            sample_rate = sample_rate.item()
        else:
            sample_rate = sample_rate.reshape(-1)[-1].item()
    return audio_data, int(sample_rate)


def compute_placeholder_count(
    audio_len_samples: int,
    *,
    sampling_rate: int,
    frame_rate: float,
) -> int:
    if audio_len_samples <= 0:
        return 0
    return compute_num_audio_input_tokens(
        audio_len_samples,
        sampling_rate=sampling_rate,
        frame_rate=frame_rate,
    )


def _stable_request_seed(
    *,
    request_id: str | None = None,
    text: str,
    task_type: str | None = None,
    ref_audio: str | None = None,
    base_seed: int = 0,
) -> int | None:
    seed_mode = ENV.tts_seed_mode
    if seed_mode in {"", "none", "off", "disabled"}:
        return None
    if seed_mode == "content":
        payload = f"{task_type or ''}\n{text or ''}\n{ref_audio or ''}"
    else:
        payload = str(request_id or "")
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=4).digest()
    salt = int.from_bytes(digest, byteorder="little", signed=False)
    return int((int(base_seed) + salt) & 0x7FFFFFFF)


class RaonServingHooks:
    """Encapsulates all Raon-specific serving behaviour."""

    def __init__(self, model_config: Any) -> None:
        self._model_config = model_config
        self._hf_config = getattr(model_config, "hf_config", None)

        self._audio_stop_token_ids: list[int] = [IM_END.id, AUDIO_END.id]

    def get_default_chat_modalities(self) -> list[str]:
        return ["text"]

    def override_output_mode(self, output_mode: str | None) -> str | None:
        return output_mode

    def should_force_audio_first_token(self, output_modalities: list[str]) -> bool:
        return "audio" in output_modalities

    def apply_task_sampling_params(
        self,
        params: SamplingParams,
        *,
        task: str,
        request: Any = None,
    ) -> None:
        """Apply task defaults from YAML, then request overrides.

        Priority: YAML default_sampling_params < task_sampling_params < request
        """
        # 1. Apply task-specific defaults from raon.yaml
        task_defaults = _get_task_sampling_params().get(task, {})
        for k, v in task_defaults.items():
            if hasattr(params, k):
                setattr(params, k, v)
            else:
                logger.warning("Unknown sampling param in YAML task_sampling_params: %s", k)

        # 2. Request overrides take precedence
        if request is not None:
            for field in (
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "min_tokens",
                "seed",
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
            ):
                value = getattr(request, field, None)
                if value is not None:
                    setattr(params, field, value)

        # 3. TTS-specific: stop tokens and audio generation constraints
        if task == "tts":
            stop_ids = list(self._audio_stop_token_ids)
            params.stop = []
            if stop_ids:
                params.stop_token_ids = stop_ids
            params.ignore_eos = False
            params.min_tokens = max(getattr(params, "min_tokens", 0) or 0, 1)

    # Keep backward-compat alias
    def prepare_audio_sampling_params(self, params: SamplingParams) -> None:
        self.apply_task_sampling_params(params, task="tts")

    @staticmethod
    def detect_chat_task(messages: list) -> str:
        """Detect Raon task type from chat message content.

        - audio + text content in user messages → speechqa
        - audio only                            → spokenqa
        - text only                             → textqa
        """
        has_audio = False
        has_text = False
        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role != "user":
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str):
                if content.strip():
                    has_text = True
                continue
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type", "")
                else:
                    t = getattr(item, "type", "")
                if t in ("audio_url", "input_audio"):
                    has_audio = True
                elif t == "text":
                    text_val = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                    if text_val and text_val.strip():
                        has_text = True
        if has_audio and has_text:
            return "speechqa"
        if has_audio:
            return "spokenqa"
        return "textqa"

    @staticmethod
    def normalize_audio_mm_item(item: Any) -> Any:
        if isinstance(item, list):
            return [RaonServingHooks.normalize_audio_mm_item(x) for x in item]
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
            return item[0]
        return item

    def normalize_audio_prompt(
        self,
        engine_prompt: Any,
        tokenizer: Any,
    ) -> None:
        if not isinstance(engine_prompt, dict):
            return
        multi_modal_data = engine_prompt.get("multi_modal_data")
        if not isinstance(multi_modal_data, dict):
            return
        if "audio" not in multi_modal_data:
            return
        multi_modal_data["audio"] = self.normalize_audio_mm_item(multi_modal_data.get("audio"))
        engine_prompt["multi_modal_data"] = multi_modal_data

        prompt_token_ids = engine_prompt.get("prompt_token_ids")
        if isinstance(prompt_token_ids, list) and all(isinstance(t, int) for t in prompt_token_ids):
            engine_prompt["prompt_token_ids"] = normalize_token_ids(prompt_token_ids)

    async def build_tts_prompt(
        self,
        text: str,
        *,
        prepend_speaker_token: bool = False,
        engine_client: Any = None,
    ) -> str:
        tokenizer = None
        if engine_client is not None:
            try:
                tokenizer = await engine_client.get_tokenizer()
            except Exception:
                pass

        builder = RaonChatTemplateBuilder(tokenizer=tokenizer or _NullTokenizer())
        result = builder.build_prompt(
            task=TaskType.TTS,
            user_content=text,
            tts_prompt_style=ENV.tts_prompt_style,
            prepend_speaker_token=prepend_speaker_token,
            append_audio_start=ENV.tts_append_audio_start,
        )
        return result.prompt_text

    def estimate_tts_min_tokens(self, text: str, *, max_tokens: int | None) -> int:
        words = [w for w in text.split() if w]
        estimated = max(6, int(len(words) * 2))
        if max_tokens is not None:
            estimated = min(estimated, max(0, int(max_tokens) - 1))
        return max(0, estimated)

    def estimate_tts_max_tokens(self, text: str, *, hard_cap: int = TTS_MAX_TOKENS_HARD_CAP) -> int:
        fixed_override = ENV.tts_fixed_max_tokens
        if fixed_override is not None:
            return max(1, min(int(hard_cap), int(fixed_override)))

        # Char-based scaling with word-based cap for spaced scripts
        # to prevent over-allocation on short text.
        char_estimate = max(96, len(text) * 20)
        words = [w for w in text.split() if w]
        n_words = len(words)
        if n_words > 0 and len(text) / n_words < 8:
            word_estimate = max(96, n_words * 16 + 64)
            char_estimate = min(char_estimate, word_estimate)
        return max(1, min(int(hard_cap), char_estimate))

    def apply_sampling_parity(
        self,
        sampling_params_list: list[Any],
        comprehension_stage_index: int,
    ) -> None:
        if not sampling_params_list:
            return
        idx = max(0, min(comprehension_stage_index, len(sampling_params_list) - 1))
        self.prepare_audio_sampling_params(sampling_params_list[idx])

    def _configured_speaker_token_id(self) -> int | None:
        hf_config = self._hf_config
        speaker_token_id = getattr(hf_config, "speaker_token_id", None)
        if isinstance(speaker_token_id, (list, tuple)):
            speaker_token_id = speaker_token_id[0] if speaker_token_id else None
        return int(speaker_token_id) if isinstance(speaker_token_id, int) else None

    def has_speaker_token(self) -> bool:
        if self._configured_speaker_token_id() is not None:
            return True
        hf_config = self._hf_config
        return getattr(hf_config, "speaker_encoder_config", None) is not None

    def resolve_chat_modalities(
        self,
        request: Any,
        engine_prompts: list,
        tokenizer: Any,
    ) -> list[str]:
        requested = getattr(request, "modalities", None)
        default = self.get_default_chat_modalities() if requested is None else None
        normalized = canonicalize_modalities(
            requested_modalities=requested,
            default_modalities=default or getattr(request, "_engine_output_modalities", None),
        )
        output_modalities = normalized or getattr(request, "_engine_output_modalities", ["text"])
        request.modalities = output_modalities

        output_mode = modalities_to_output_mode(normalized)
        output_mode = self.override_output_mode(output_mode)
        force_audio = self.should_force_audio_first_token(output_modalities)

        for ep in engine_prompts:
            attach_output_mode_additional_information(ep, output_mode)
            if force_audio:
                ai = ep.get("additional_information")
                if not isinstance(ai, dict):
                    ai = {}
                ai["force_audio_first_token"] = [True]
                ep["additional_information"] = ai
            self.normalize_audio_prompt(ep, tokenizer)

        return output_modalities

    async def _build_icl_tts_prompt(
        self,
        *,
        target_text: str,
        ref_text: str,
        prepend_speaker_token: bool = True,
        engine_client: Any = None,
    ) -> str:
        """Build ICL prompt for voice cloning.

        Prompt layout::

            <|im_start|>user
            [speaker_embed] Speak the following text:
            {ref_text} {target_text}<|im_end|>
            <|im_start|>assistant
            <|audio_start|>[output_embed x N_ref]

        No trailing ``<audio_start>`` — the model continues generating
        from the last ref frame.
        """
        user_content = f"Speak the following text:\n{ref_text} {target_text}"
        if prepend_speaker_token:
            speaker_token = SPEAKER_EMBEDDING_PLACEHOLDER.text
            if speaker_token is not None and not user_content.startswith(speaker_token):
                user_content = f"{speaker_token}{user_content}"

        assistant_content = f"{AUDIO_START_TOKEN}{AUDIO_OUTPUT_PAD_TOKEN}"

        tokenizer = None
        if engine_client is not None:
            try:
                tokenizer = await engine_client.get_tokenizer()
            except Exception:
                pass

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        prompt: str | None = None
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if isinstance(prompt, str):
                    # Remove trailing <|im_end|> — model continues from here.
                    prompt = re.sub(r"<\|im_end\|>\s*$", "", prompt)
            except Exception:
                prompt = None

        if prompt is None:
            prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}"

        return prompt

    @staticmethod
    def _is_icl_request(request: Any) -> bool:
        """Return True if request is an ICL voice cloning request."""
        task_type = getattr(request, "task_type", None)
        if task_type is None:
            ref_audio = getattr(request, "ref_audio", None)
            if ref_audio is not None:
                task_type = "Base"
        if task_type != "Base":
            return False
        x_vector_only = bool(getattr(request, "x_vector_only_mode", False))
        if x_vector_only:
            return False
        ref_audio = getattr(request, "ref_audio", None)
        ref_text = getattr(request, "ref_text", None)
        return bool(isinstance(ref_audio, str) and ref_audio.strip() and isinstance(ref_text, str) and ref_text.strip())

    @staticmethod
    def validate_request(request: Any) -> str | None:
        """Validate Raon request. Returns error message or None."""
        # Auto-infer Base task when ref_audio, ref_text, or speaker_embedding is provided.
        has_speaker_embedding = getattr(request, "speaker_embedding", None) is not None
        if request.task_type is None and (
            request.ref_audio is not None or request.ref_text is not None or has_speaker_embedding
        ):
            request.task_type = "Base"

        task_type = request.task_type

        if task_type == "Base":
            if request.ref_audio is None and not has_speaker_embedding:
                return "Base task requires 'ref_audio' or 'speaker_embedding' for voice cloning"
            ref_audio = request.ref_audio
            if ref_audio is not None:
                is_valid_scheme = (
                    ref_audio.startswith(("http://", "https://"))
                    or ref_audio.startswith("data:")
                    or ref_audio.startswith("file://")
                )
                if not is_valid_scheme:
                    return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
                if not request.x_vector_only_mode:
                    if not request.ref_text or not request.ref_text.strip():
                        request.x_vector_only_mode = True

        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < 1:
                return "max_new_tokens must be at least 1"
            if request.max_new_tokens > 8192:
                return "max_new_tokens cannot exceed 8192"

        return None

    @staticmethod
    async def _resolve_ref_audio(ref_audio_str: str, model_config: Any) -> tuple:
        """Resolve ref_audio URI to (ndarray, sample_rate) using vLLM MediaConnector."""

        from vllm.multimodal.media import MediaConnector

        connector = MediaConnector(
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        wav_np, sr = await connector.fetch_audio_async(ref_audio_str)
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=-1)
        return wav_np, int(sr)
