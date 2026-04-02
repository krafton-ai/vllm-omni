# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai.inference_context import inference_ctx
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.transformers_utils.configs.raon import (
    AUDIO_SAMPLE_RATE,
    ENV,
    TTS_MAX_TOKENS_HARD_CAP,
)
from vllm_omni.model_executor.models.raon.raon_multimodal import compute_num_audio_input_tokens
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END,
    AUDIO_INPUT_PAD_TOKEN,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_PLACEHOLDER_SEQ,
    AUDIO_START,
    AUDIO_START_TOKEN,
    IM_END,
    RaonChatTemplateBuilder,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    TaskType,
    align_tokenizer,
    filter_audio_placeholder_text,
    normalize_token_ids,
)
from vllm_omni.model_executor.models.registry import create_serving_hooks

logger = init_logger(__name__)


class _NullTokenizer:
    """Sentinel used when no real tokenizer is available."""


MODALITY_TEXT = "text"
MODALITY_AUDIO = "audio"

OUTPUT_MODE_TEXT_ONLY = "text_only"
OUTPUT_MODE_AUDIO_ONLY = "audio_only"
OUTPUT_MODE_TEXT_AND_AUDIO = "text_and_audio"

_GLOBAL_TOP_K = 20
_TASK_DEFAULTS: dict[str, dict[str, float | int]] = {
    "tts": {
        "temperature": 1.2,
        "top_p": 0.8,
        "top_k": 50,
        "max_tokens": 256,
    },
    "stt": {
        "temperature": 0.2,
        "max_tokens": 512,
    },
    "speechqa": {
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "max_tokens": 2048,
    },
    "spokenqa": {
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "max_tokens": 2048,
    },
}

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
        sample_rate = sample_rate.item() if sample_rate.numel() == 1 else sample_rate.reshape(-1)[-1].item()
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


class _RaonHooksMixin:
    """Mixin that lazily resolves serving hooks for Raon models."""

    _serving_hooks: object | None = None
    _serving_hooks_resolved: bool = False

    @property
    def _hooks(self) -> object | None:
        if not self._serving_hooks_resolved:
            model_config = getattr(self, "model_config", None)
            if model_config is None:
                model_config = getattr(
                    getattr(self, "engine_client", None),
                    "model_config",
                    None,
                )
            hf_config = getattr(model_config, "hf_config", None)
            model_type = str(getattr(hf_config, "model_type", "")).lower()
            if model_type:
                self._serving_hooks = create_serving_hooks(model_type, model_config)
            self._serving_hooks_resolved = True
        return self._serving_hooks


class RaonOpenAIServingChat(_RaonHooksMixin, OmniOpenAIServingChat):
    """Chat serving with Raon-specific modality and audio handling."""

    @staticmethod
    def _inject_audio_placeholder(messages):
        placeholder = AUDIO_PLACEHOLDER_SEQ
        updated_messages = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if not isinstance(content, list):
                updated_messages.append(message)
                continue

            has_audio = any(
                isinstance(item, dict) and item.get("type") in {"audio_url", "input_audio"} for item in content
            )
            if not has_audio:
                updated_messages.append(message)
                continue

            has_placeholder = any(
                isinstance(item, dict) and item.get("type") == "text" and placeholder in str(item.get("text", ""))
                for item in content
            )
            if has_placeholder:
                updated_messages.append(message)
                continue

            new_content: list[Any] = []
            injected = False
            for item in content:
                if not injected and isinstance(item, dict) and item.get("type") == "text":
                    new_item = dict(item)
                    new_item["text"] = f"{placeholder}{new_item.get('text', '')}"
                    new_content.append(new_item)
                    injected = True
                else:
                    new_content.append(item)

            if not injected:
                new_content.insert(0, {"type": "text", "text": placeholder})

            if isinstance(message, dict):
                new_message = dict(message)
                new_message["content"] = new_content
            else:
                new_message = message.model_copy(deep=True) if hasattr(message, "model_copy") else message
                setattr(new_message, "content", new_content)
            updated_messages.append(new_message)

        return updated_messages

    async def _preprocess_chat(
        self,
        request,
        messages,
        default_template,
        default_template_content_format,
        default_template_kwargs=None,
        tool_dicts=None,
        tool_parser=None,
        renderer=None,
        add_generation_prompt=True,
        continue_final_message=False,
        documents=None,
        add_special_tokens=False,
    ):
        messages = self._inject_audio_placeholder(messages)
        tokenizer = None
        if renderer is not None:
            try:
                tokenizer = renderer.get_tokenizer()
            except Exception:
                pass
            align_tokenizer(tokenizer)
        conversation, engine_prompts = await super()._preprocess_chat(
            request=request,
            messages=messages,
            default_template=default_template,
            default_template_content_format=default_template_content_format,
            default_template_kwargs=default_template_kwargs,
            tool_dicts=tool_dicts,
            tool_parser=tool_parser,
            renderer=renderer,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            documents=documents,
            add_special_tokens=add_special_tokens,
        )
        hooks = self._hooks
        if hooks is not None:
            hooks.resolve_chat_modalities(request, engine_prompts, tokenizer)
        return conversation, engine_prompts

    def _filter_text_content(self, text):
        return filter_audio_placeholder_text(text)

    def _extract_audio_data_and_sample_rate(self, mm_output):
        return extract_audio_data_and_sample_rate(mm_output)


class RaonOpenAIServingSpeech(_RaonHooksMixin, OmniOpenAIServingSpeech):
    """Speech serving with Raon-specific TTS prompt and sampling logic."""

    def _get_comprehension_stage_index(self) -> int:
        for idx, stage in enumerate(self.engine_client.stage_configs):
            if stage.is_comprehension:
                return idx
        raise ValueError("No comprehension stage found in stage_configs")

    def _get_comprehension_sampling_params(
        self,
        sampling_params_list: list[SamplingParams | dict[str, Any]],
    ) -> SamplingParams | dict[str, Any] | None:
        if not sampling_params_list:
            return None
        idx = self._get_comprehension_stage_index()
        idx = max(0, min(idx, len(sampling_params_list) - 1))
        return sampling_params_list[idx]

    async def _prepare_speech_prompt(self, request, tts_params, request_id):
        hooks = self._hooks
        if tts_params:
            # TTS model path — delegate to base
            return await super()._prepare_speech_prompt(request, tts_params, request_id)

        # TTS path
        if hooks is not None:
            additional_information: dict[str, Any] = {}
            additional_information["global_request_id"] = [str(request_id)]
            additional_information["source_text"] = [str(getattr(request, "input", "") or "")]
            if request.ref_audio is not None:
                additional_information["speaker_ref_audio"] = [request.ref_audio]
            has_speaker = bool(request.ref_audio is not None and hooks.has_speaker_token())
            is_icl = hooks._is_icl_request(request)
            additional_information["force_audio_first_token"] = [True]
            logger.info("TTS %s: icl=%s speaker=%s", request_id, is_icl, has_speaker)

            if is_icl:
                # ICL path: build prompt with multimodal ref audio prefill
                additional_information["icl_mode"] = [True]
                additional_information["continuation_silence_frames"] = [ENV.continuation_silence_frames]
                additional_information["source_ref_text"] = [str(request.ref_text or "")]

                prompt_text = await hooks._build_icl_tts_prompt(
                    target_text=request.input,
                    ref_text=request.ref_text,
                    prepend_speaker_token=has_speaker,
                    engine_client=self.engine_client,
                )
                prompt: dict[str, Any] = {"prompt": prompt_text}

                # Resolve ref_audio to waveform for multimodal pipeline
                try:
                    wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
                    wav_np = np.asarray(wav_list, dtype=np.float32) if isinstance(wav_list, list) else wav_list
                    prompt["multi_modal_data"] = {"audio": [(wav_np, sr)]}
                except Exception as exc:
                    logger.warning("Failed to resolve ref_audio: %s", exc)

                if request.max_new_tokens is None:
                    try:
                        request.max_new_tokens = int(
                            hooks.estimate_tts_max_tokens(
                                request.input,
                            )
                        )
                    except Exception:
                        request.max_new_tokens = TTS_MAX_TOKENS_HARD_CAP
            else:
                # Standard / x_vector_only path
                prompt = {
                    "prompt": await hooks.build_tts_prompt(
                        request.input,
                        prepend_speaker_token=has_speaker,
                        engine_client=self.engine_client,
                    )
                }
                if request.max_new_tokens is None:
                    try:
                        request.max_new_tokens = int(
                            hooks.estimate_tts_max_tokens(
                                request.input,
                                hard_cap=TTS_MAX_TOKENS_HARD_CAP,
                            )
                        )
                    except Exception:
                        request.max_new_tokens = TTS_MAX_TOKENS_HARD_CAP

            prompt["additional_information"] = additional_information
            inference_ctx.update_ctx(
                str(request_id),
                {
                    "output_modalities": ["audio"],
                    "output_mode": "audio_only",
                    "force_audio_first_token": True,
                    "speaker_ref_audio": request.ref_audio is not None,
                    "icl_mode": is_icl,
                    "tts_max_new_tokens": request.max_new_tokens,
                    "source_text_len": len(str(getattr(request, "input", "") or "")),
                },
            )
            return prompt

        # Generic fallback
        return {"prompt": request.input}

    def _attach_speech_output_mode(self, prompt):
        hooks = self._hooks
        output_mode = modalities_to_output_mode([MODALITY_AUDIO])
        if hooks is not None:
            output_mode = hooks.override_output_mode(output_mode)
        attach_output_mode_additional_information(prompt, output_mode)

    async def _finalize_speech_sampling_params(self, sampling_params_list, request, request_id):
        hooks = self._hooks
        if hooks is None:
            return
        self._apply_sampling_parity(sampling_params_list)
        params = self._get_comprehension_sampling_params(sampling_params_list)
        if params is None:
            return
        try:
            preferred_temperature = ENV.tts_temperature
            preferred_top_k = ENV.tts_top_k
            preferred_top_p = ENV.tts_top_p
            preferred_repetition_penalty = ENV.tts_repetition_penalty
            if isinstance(params, SamplingParams):
                params.temperature = preferred_temperature
                params.top_k = preferred_top_k
                params.top_p = preferred_top_p
                if hasattr(params, "repetition_penalty"):
                    params.repetition_penalty = preferred_repetition_penalty
            elif isinstance(params, dict):
                params["temperature"] = preferred_temperature
                params["top_k"] = preferred_top_k
                params["top_p"] = preferred_top_p
                params["repetition_penalty"] = preferred_repetition_penalty
        except Exception as exc:
            logger.warning("Failed to set preferred sampling defaults: %s", exc)
        try:
            max_tokens = getattr(request, "max_new_tokens", None)
            if max_tokens is not None:
                if isinstance(params, SamplingParams):
                    params.max_tokens = int(max_tokens)
                elif isinstance(params, dict):
                    params["max_tokens"] = int(max_tokens)
        except Exception as exc:
            logger.warning("Failed to set max_tokens: %s", exc)
        try:
            min_tokens = hooks.estimate_tts_min_tokens(
                request.input,
                max_tokens=request.max_new_tokens,
            )
            if isinstance(params, SamplingParams):
                params.min_tokens = max(int(getattr(params, "min_tokens", 0)), int(min_tokens))
                base_seed = int(getattr(params, "seed", 0) or 0)
                derived_seed = _stable_request_seed(
                    request_id=str(request_id),
                    text=str(getattr(request, "input", "") or ""),
                    task_type=str(getattr(request, "task_type", "") or ""),
                    ref_audio=str(getattr(request, "ref_audio", "") or ""),
                    base_seed=base_seed,
                )
                params.seed = derived_seed
            elif isinstance(params, dict):
                params["min_tokens"] = max(int(params.get("min_tokens", 0) or 0), int(min_tokens))
                base_seed = int(params.get("seed", 0) or 0)
                derived_seed = _stable_request_seed(
                    request_id=str(request_id),
                    text=str(getattr(request, "input", "") or ""),
                    task_type=str(getattr(request, "task_type", "") or ""),
                    ref_audio=str(getattr(request, "ref_audio", "") or ""),
                    base_seed=base_seed,
                )
                if derived_seed is None:
                    params.pop("seed", None)
                else:
                    params["seed"] = derived_seed
        except Exception as exc:
            logger.warning("Failed to set min_tokens: %s", exc)

    def _build_sampling_params_list(self, request) -> list:
        import copy

        del request
        default = getattr(self.engine_client, "default_sampling_params_list", [])
        params_list = []
        for p in default:
            if isinstance(p, SamplingParams):
                params_list.append(p.clone())
            elif isinstance(p, dict):
                params_list.append(SamplingParams(**p))
            else:
                params_list.append(copy.deepcopy(p))
        return params_list

    async def create_speech(self, request, raw_request=None):
        import asyncio

        from fastapi.responses import Response, StreamingResponse
        from vllm.utils import random_uuid

        from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"
        try:
            if self._is_tts:
                # Qwen3-TTS path — delegate to base
                return await super().create_speech(request, raw_request)

            # TTS path
            prompt = await self._prepare_speech_prompt(request, {}, request_id)
            self._attach_speech_output_mode(prompt)

            sampling_params_list = self._build_sampling_params_list(request)
            await self._finalize_speech_sampling_params(
                sampling_params_list,
                request,
                request_id,
            )

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            if request.stream:
                response_format = (request.response_format or "wav").lower()
                if response_format not in ("pcm", "wav"):
                    return self.create_error_response(
                        f"Streaming only supports 'pcm' and 'wav'. Got '{response_format}'."
                    )
                media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
                return StreamingResponse(
                    self._generate_audio_chunks(generator, request_id, response_format),
                    media_type=media_type,
                )

            final_output = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return self.create_error_response("No output generated.")

            audio_output, audio_key = self._extract_audio_output(final_output)
            if audio_key is None:
                return self.create_error_response("Model did not produce audio.")

            audio_tensor = audio_output[audio_key]
            sr_raw = audio_output.get("sr", AUDIO_SAMPLE_RATE)
            sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
            sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

            if isinstance(audio_tensor, list):
                audio_tensor = torch.cat(audio_tensor, dim=-1)
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
                stream_format=request.stream_format,
                base64_encode=False,
            )
            audio_response = self.create_audio(audio_obj)
            return Response(
                content=audio_response.audio_data,
                media_type=audio_response.media_type,
            )

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)
        except Exception as e:
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")

    # -- internal helpers --------------------------------------------------

    def _apply_sampling_parity(
        self,
        sampling_params_list: list[SamplingParams | dict[str, Any]],
    ) -> None:
        hooks = self._hooks
        if hooks is not None:
            hooks.apply_sampling_parity(
                sampling_params_list,
                self._get_comprehension_stage_index(),
            )
            return
        if not sampling_params_list:
            return
        idx = self._get_comprehension_stage_index()
        idx = max(0, min(idx, len(sampling_params_list) - 1))
        params = sampling_params_list[idx]
        if isinstance(params, SamplingParams):
            params.stop = []
            params.stop_token_ids = []
            params.ignore_eos = False
            params.min_tokens = max(getattr(params, "min_tokens", 0) or 0, 1)
        elif isinstance(params, dict):
            params["stop"] = []
            params["stop_token_ids"] = []
            params["ignore_eos"] = False
            params["min_tokens"] = max(int(params.get("min_tokens", 0) or 0), 1)


class RaonServingHooks:
    """Encapsulates all Raon-specific serving behaviour."""

    def __init__(self, model_config: Any) -> None:
        self._model_config = model_config
        self._hf_config = getattr(model_config, "hf_config", None)

        self._audio_stop_token_ids: list[int] = [IM_END.id, AUDIO_END.id]
        self._voice_cache_manager: Any | None = None
        self._voice_cache_resolved: bool = False

    def _get_voice_cache_manager(self) -> Any | None:
        """Lazily create a RaonVoiceCacheManager if voice samples dir is available."""
        if not self._voice_cache_resolved:
            self._voice_cache_resolved = True
            samples_dir = os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_samples")
            if samples_dir and os.path.isdir(samples_dir):
                try:
                    from vllm_omni.model_executor.models.raon.raon_voice_cache import (
                        RaonVoiceCacheManager,
                    )

                    self._voice_cache_manager = RaonVoiceCacheManager(samples_dir)
                except Exception as exc:
                    logger.warning("Failed to init voice cache manager: %s", exc)
        return self._voice_cache_manager

    def get_default_chat_modalities(self) -> list[str]:
        return ["text"]

    def override_output_mode(self, output_mode: str | None) -> str | None:
        return output_mode

    def should_force_audio_first_token(self, output_modalities: list[str]) -> bool:
        return "audio" in output_modalities

    def prepare_audio_sampling_params(self, params: Any) -> None:
        stop_ids = list(self._audio_stop_token_ids)

        if isinstance(params, SamplingParams):
            params.stop = []
            if stop_ids:
                params.stop_token_ids = stop_ids
            params.ignore_eos = False
            params.min_tokens = max(getattr(params, "min_tokens", 0) or 0, 1)
            params.temperature = 1.2
            params.top_p = 0.8
            params.top_k = _GLOBAL_TOP_K
        elif isinstance(params, dict):
            params["stop"] = []
            if stop_ids:
                params["stop_token_ids"] = stop_ids
            params["ignore_eos"] = False
            params["min_tokens"] = max(int(params.get("min_tokens", 0) or 0), 1)
            params["temperature"] = 1.2
            params["top_p"] = 0.8
            params["top_k"] = _GLOBAL_TOP_K

    @staticmethod
    def _param_get(params: Any, key: str, default: Any = None) -> Any:
        if isinstance(params, SamplingParams):
            return getattr(params, key, default)
        if isinstance(params, dict):
            return params.get(key, default)
        return default

    @staticmethod
    def _param_set(params: Any, key: str, value: Any) -> None:
        if isinstance(params, SamplingParams):
            setattr(params, key, value)
        elif isinstance(params, dict):
            params[key] = value

    def apply_task_defaults(self, params: Any, task_type: str) -> None:
        defaults = _TASK_DEFAULTS.get(str(task_type).lower())
        if defaults is None:
            return

        baseline = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        }
        for key, value in defaults.items():
            current = self._param_get(params, key, None)
            if current is None:
                self._param_set(params, key, value)
                continue
            if key in baseline and float(current) == float(baseline[key]):
                self._param_set(params, key, value)

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

        words = [w for w in text.split() if w]
        estimated = int(len(words) * 8 + 32)
        estimated = max(48, estimated)
        estimated = min(int(hard_cap), estimated)
        return max(1, estimated)

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

    def resolve_chat_modalities(self, request: Any, engine_prompts: list, tokenizer: Any) -> list[str]:
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

    def filter_text_content(self, text: str | None) -> str | None:
        return filter_audio_placeholder_text(text)

    def extract_audio_data(self, mm_output: dict) -> tuple[Any, int]:
        return extract_audio_data_and_sample_rate(mm_output)

    async def _build_icl_tts_prompt(
        self,
        *,
        target_text: str,
        ref_text: str,
        prepend_speaker_token: bool = True,
        engine_client: Any = None,
    ) -> str:
        """Build ICL prompt for voice cloning.

        Prompt layout after multimodal expansion::

            <|im_start|>user
            [speaker_embed] Speak the following text:
            {ref_text} {target_text}<|im_end|>
            <|im_start|>assistant
            <|audio_start|>[output_embed x N_ref]
            ^--- Mimi codec output embeddings ---^  model continues from here

        ``AUDIO_OUTPUT_PAD_TOKEN`` routes through Mimi codec →
        quantizer.decode → output_adaptor (same path as training).
        No trailing ``<audio_start>`` — the model continues generating
        directly from the last ref frame (a second ``<audio_start>``
        is out-of-distribution and causes restart).
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
            prompt = (
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_content}"
            )

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
        return bool(
            isinstance(ref_audio, str) and ref_audio.strip()
            and isinstance(ref_text, str) and ref_text.strip()
        )

    @staticmethod
    def validate_request(request: Any) -> str | None:
        """Validate Raon TTS request. Returns error message or None."""
        # Auto-infer Base task when ref_audio or ref_text is provided.
        if request.task_type is None and (request.ref_audio is not None or request.ref_text is not None):
            request.task_type = "Base"

        task_type = request.task_type

        if task_type == "Base":
            if request.ref_audio is None:
                return "Base task requires 'ref_audio' for voice cloning"
            ref_audio = request.ref_audio
            if not (ref_audio.startswith(("http://", "https://")) or ref_audio.startswith("data:") or ref_audio.startswith("file://")):
                return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
            if not request.x_vector_only_mode:
                if not request.ref_text or not request.ref_text.strip():
                    return (
                        "Base task requires non-empty 'ref_text' (transcript of "
                        "the reference audio) unless 'x_vector_only_mode' is enabled"
                    )

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

    async def prepare_speech(
        self,
        request: Any,
        engine_client: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], list | None]:
        import copy

        from vllm.utils import random_uuid

        validation_error = self.validate_request(request)
        if validation_error:
            raise ValueError(validation_error)

        request_id = f"speech-{random_uuid()}"
        # --- Detect ICL / Base task mode ---
        task_type = getattr(request, "task_type", None)
        if task_type is None and request.ref_audio is not None:
            task_type = "Base"
        x_vector_only = bool(getattr(request, "x_vector_only_mode", False))
        icl_mode = (task_type == "Base") and not x_vector_only

        additional_information: dict[str, Any] = {
            "global_request_id": [str(request_id)],
            "source_text": [str(getattr(request, "input", "") or "")],
            "force_audio_first_token": [True],
        }
        has_speaker = bool(request.ref_audio is not None and self.has_speaker_token())

        if request.ref_audio is not None:
            additional_information["speaker_ref_audio"] = [request.ref_audio]

        if icl_mode and request.ref_audio is not None and request.ref_text:
            # --- ICL path: multimodal prefill with ref audio ---
            additional_information["icl_mode"] = [True]
            additional_information["continuation_silence_frames"] = [ENV.continuation_silence_frames]
            additional_information["source_ref_text"] = [str(request.ref_text)]

            # Check voice cache for pre-computed speaker embedding.
            voice_name = getattr(request, "voice", None)
            cache_mgr = self._get_voice_cache_manager()
            if voice_name and cache_mgr is not None:
                cached = cache_mgr.load_cached_voice_prompt(voice_name)
                if cached is not None:
                    additional_information["cached_spk_embedding"] = [cached.ref_spk_embedding]
                    if cached.ref_codec_codes is not None:
                        additional_information["cached_ref_codec_codes"] = [cached.ref_codec_codes]
                        if cached.ref_codec_codes_mask is not None:
                            additional_information["cached_ref_codec_codes_mask"] = [cached.ref_codec_codes_mask]

            model_config = getattr(engine_client, "model_config", None)
            wav_np, sr = await self._resolve_ref_audio(request.ref_audio, model_config)
            prompt_text = await self._build_icl_tts_prompt(
                target_text=request.input,
                ref_text=request.ref_text,
                prepend_speaker_token=has_speaker,
                engine_client=engine_client,
            )

            prompt: dict[str, Any] = {
                "prompt": prompt_text,
                "multi_modal_data": {"audio": [(wav_np, sr)]},
            }

        else:
            # --- Standard / x_vector_only path (unchanged) ---
            if task_type == "Base":
                additional_information["icl_mode"] = [False]
                additional_information["x_vector_only_mode"] = [True]

            prompt = {
                "prompt": await self.build_tts_prompt(
                    request.input,
                    prepend_speaker_token=has_speaker,
                    engine_client=engine_client,
                )
            }

        prompt["additional_information"] = additional_information

        output_mode = modalities_to_output_mode([MODALITY_AUDIO])
        output_mode = self.override_output_mode(output_mode)
        attach_output_mode_additional_information(prompt, output_mode)

        if request.max_new_tokens is None:
            try:
                if icl_mode:
                    request.max_new_tokens = int(
                        self.estimate_tts_max_tokens(
                            request.input,
                        )
                    )
                else:
                    request.max_new_tokens = int(
                        self.estimate_tts_max_tokens(
                            request.input,
                            hard_cap=TTS_MAX_TOKENS_HARD_CAP,
                        )
                    )
            except Exception:
                request.max_new_tokens = 512

        default = list(getattr(engine_client, "default_sampling_params_list", []))
        params_list: list[Any] = []
        for p in default:
            if isinstance(p, SamplingParams):
                params_list.append(p.clone())
            elif isinstance(p, dict):
                params_list.append(SamplingParams(**p))
            else:
                params_list.append(copy.deepcopy(p))

        if params_list:
            comp_idx = self._find_comprehension_index(engine_client)
            idx = max(0, min(comp_idx, len(params_list) - 1))
            self.prepare_audio_sampling_params(params_list[idx])

            params = params_list[idx]
            if isinstance(params, SamplingParams):
                params.temperature = ENV.tts_temperature
                params.top_k = ENV.tts_top_k
                params.top_p = ENV.tts_top_p
                if hasattr(params, "repetition_penalty"):
                    params.repetition_penalty = ENV.tts_repetition_penalty
                if request.max_new_tokens is not None:
                    params.max_tokens = int(request.max_new_tokens)
                min_tokens = self.estimate_tts_min_tokens(
                    request.input,
                    max_tokens=request.max_new_tokens,
                )
                params.min_tokens = max(int(getattr(params, "min_tokens", 0)), int(min_tokens))

                derived_seed = _stable_request_seed(
                    request_id=str(request_id),
                    text=str(getattr(request, "input", "") or ""),
                    task_type=str(getattr(request, "task_type", "") or ""),
                    ref_audio=str(getattr(request, "ref_audio", "") or ""),
                    base_seed=int(getattr(params, "seed", 0) or 0),
                )
                if derived_seed is not None:
                    params.seed = derived_seed

        return prompt, {}, params_list or None

    @staticmethod
    def _find_comprehension_index(engine_client: Any) -> int:
        for idx, stage in enumerate(engine_client.stage_configs):
            if stage.is_comprehension:
                return idx
        return 0
