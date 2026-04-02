# SPDX-License-Identifier: Apache-2.0

"""Raon vLLM model: AR thinker + talker with audio codec integration."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional
from torch import nn
from transformers import AutoTokenizer
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.raon import (
    ENV,
    REQUEST_STATE_CLEANUP_KEYS,
    REQUEST_STATE_CLEANUP_PREFIXES,
    SpeakerEncoderConfig,
    coerce_speaker_encoder_config,
    get_mimi_frame_rate,
)
from vllm_omni.model_executor.models.raon.raon_multimodal import (
    RaonDummyInputsBuilder,
    RaonMultiModalProcessor,
    RaonProcessingInfo,
    compute_samples_per_frame,
    flatten_audio_embeddings,
    normalize_audio_waveforms_and_lengths,
    scatter_audio_input_embeddings,
    strip_raon_audio_markers,
)
from vllm_omni.model_executor.models.raon.raon_adaptors import (
    EmbeddingAdaptor,
    ThinkerToTalkerProjection,
)
from vllm_omni.model_executor.models.raon.raon_audio_encoder import Qwen3OmniAuTWrapper
from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import StreamingMimiModel
from vllm_omni.model_executor.models.raon.raon_code_predictor import (
    RaonCodePredictor,
    RepetitionAwareSampler,
)
from vllm_omni.model_executor.models.raon.raon_speaker_encoder import (
    PretrainedSpeakerEncoder,
    build_speaker_encoder,
    compute_speaker_embeds,
    load_speaker_ref_audio,
    normalize_speaker_ref_audio,
)
from vllm_omni.model_executor.models.raon.raon_utils import (
    coerce_optional_int,
    collapse_exact_repeated_codec_snapshot,
    module_device,
    module_dtype,
    normalize_runtime_request_id,
    unwrap_singleton_list,
)
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END_TOKEN,
    AUDIO_INPUT_PAD_TOKEN,
    AUDIO_PLACEHOLDER_SEQ,
    AUDIO_START_TOKEN,
    align_tokenizer,
    resolve_speaker_token_id,
)

logger = init_logger(__name__)


@dataclass
class AudioDecodeState:
    pending_audio_codes: torch.Tensor | None = None
    forced_audio_bootstrap_done: bool = False
    is_generating_audio: bool = False
    audio_step_index: int = 0
    continuation_silence_frames: int = 0


@dataclass
class _DeferredPreprocessEntry:
    """Holds state for a request whose ``get_audio_output_embeds`` call was deferred."""

    input_embeds: torch.Tensor
    audio_positions: torch.Tensor
    full_codes: torch.Tensor
    update_dict: dict[str, Any]
    req_state: AudioDecodeState | None
    info_dict: dict[str, Any]


@MULTIMODAL_REGISTRY.register_processor(
    RaonMultiModalProcessor,
    info=RaonProcessingInfo,
    dummy_inputs=RaonDummyInputsBuilder,
)
class RaonModel(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    CustomProcessMixin,
):
    """Raon wrapper supporting Stage-0 AR."""

    _MAX_HIDDEN_QUEUE_DEPTH = 128
    supported_languages = ISO639_1_SUPPORTED_LANGS
    request_state_cleanup_keys = REQUEST_STATE_CLEANUP_KEYS
    request_state_cleanup_prefixes = REQUEST_STATE_CLEANUP_PREFIXES
    gpu_retained_buffer_keys: set[str] = {"duplex_prev_hidden", "speaker_embeds"}
    inject_per_request_metadata: bool = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER_SEQ
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        # HF sub-config handles and omni pipeline flags.
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        text_config = self.config.text_model_config
        talker_config = self.config.talker_config
        code_predictor_config = self.config.code_predictor_config
        audio_encoder_config = self.config.audio_encoder_config
        audio_tokenizer_config = self.config.audio_tokenizer_config
        input_adaptor_config = self.config.input_adaptor_config
        output_adaptor_config = self.config.output_adaptor_config

        self.model_stage = "stage0"
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.requires_raw_input_tokens = True  # audio placeholder detection

        dtype = vllm_config.model_config.dtype
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        self.dtype = dtype

        # Sizes, audio token ids, Mimi timing, thinker hook layer.
        self.audio_output_token_id = int(self.config.audio_output_token_id)
        self.audio_input_token_id = int(self.config.audio_input_token_id)

        self.hidden_size = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.num_thinker_layers = int(text_config.num_hidden_layers)
        self.codebook_size = int(audio_tokenizer_config.codebook_size)
        self.sampling_rate = int(audio_tokenizer_config.sampling_rate)
        self.num_code_groups = int(code_predictor_config.num_code_groups)

        self.accept_hidden_layer = self.config.accept_hidden_layer_resolved()
        self.lm_head_layer_index = self.accept_hidden_layer
        self.audio_lm_head_vocab_size = self.codebook_size + 1
        
        self.frame_rate = get_mimi_frame_rate(audio_tokenizer_config)
        self.samples_per_frame = compute_samples_per_frame(
            sampling_rate=self.sampling_rate,
            frame_rate=self.frame_rate,
        )

        audio_hidden_size = int(talker_config.hidden_size)
        pp_last = get_pp_group().is_last_rank

        projection_mode = getattr(self.config, "thinker_to_talker_projection_mode", "mlp")
        projection_intermediate_size = getattr(self.config, "thinker_to_talker_intermediate_size", None)
        if projection_mode == "mlp" and projection_intermediate_size is None:
            projection_intermediate_size = getattr(talker_config, "intermediate_size", None)

        if float(input_adaptor_config.output_time_scale) != 1.0:
            raise NotImplementedError("Only `output_time_scale == 1` is supported.")

        # vLLM configs for Qwen3 thinker / talker stacks.
        text_vllm_config = vllm_config.with_hf_config(text_config, architectures=["Qwen3ForCausalLM"])
        talker_vllm_config = vllm_config.with_hf_config(talker_config, architectures=["Qwen3ForCausalLM"])

        # Per-request decode state, hidden queues, tokenizer slots.
        self._ras = RepetitionAwareSampler()
        self._deferred_preprocess: list[_DeferredPreprocessEntry] = []
        self._audio_decode_state: dict[str, AudioDecodeState] = {}
        self._thinker_hidden_queue: deque[torch.Tensor] = deque()
        self._talker_hidden_for_logits_queue: deque[torch.Tensor] = deque()
        self._talker_hidden_for_postprocess_queue: deque[torch.Tensor] = deque()
        self._runtime_info_queue: deque[list[dict[str, Any]]] = deque()
        self._cached_silence_codes: torch.Tensor | None = None
        self._N_ICL_SILENCE_FRAMES: int = 2

        self.speaker_token_id: int | None = None
        self.speaker_encoder: PretrainedSpeakerEncoder | None = None
        self.is_pretrained_speaker_encoder = False
        self.eos_token_id: int | None = None
        self._audio_only_allowed_text_token_ids: tuple[int, ...] = ()
        self._tokenizer: Any | None = None
        self._tokenizer_len: int | None = None
        self.audio_end_token_id: int | None = None

        def _prefixed(name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        # Thinker/talker backbones, LM heads, Mimi, adaptors, code predictor.
        self.text_model = Qwen3Model(vllm_config=text_vllm_config, prefix=_prefixed("text_model"))
        self.talker = Qwen3Model(
            vllm_config=talker_vllm_config,
            prefix=_prefixed("talker"),
        )
        if hasattr(self.talker, "embed_tokens"):
            del self.talker.embed_tokens
            self.talker.embed_tokens = None  # type: ignore[assignment]

        if pp_last:
            self.lm_head = ParallelLMHead(
                self.vocab_size,
                int(text_config.hidden_size),
                quant_config=vllm_config.quant_config,
                prefix=_prefixed("lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.vocab_size)

        self.audio_lm_head = nn.Linear(
            audio_hidden_size,
            self.audio_lm_head_vocab_size,
            bias=False,
            dtype=self.dtype,
        )
        self.proj_code = nn.Linear(
            audio_hidden_size,
            int(code_predictor_config.hidden_size),
            bias=bool(getattr(self.config, "proj_code_bias", False)),
            dtype=self.dtype,
        )
        self.thinker_to_talker_proj = ThinkerToTalkerProjection(
            thinker_hidden_size=int(text_config.hidden_size),
            talker_hidden_size=audio_hidden_size,
            intermediate_size=projection_intermediate_size,
            mode=projection_mode,
            use_norm=bool(getattr(self.config, "thinker_to_talker_pre_norm", False)),
            rms_norm_eps=float(getattr(text_config, "rms_norm_eps", 1e-6)),
            dtype=self.dtype,
        )
        self.proj_speaker_code: nn.Linear | None = None
        if bool(getattr(self.config, "speaker_embedding_to_code_predictor", False)):
            self.proj_speaker_code = nn.Linear(
                audio_hidden_size,
                int(code_predictor_config.hidden_size),
                bias=False,
                dtype=self.dtype,
            )

        self.audio_encoder = self._build_audio_encoder(audio_encoder_config)
        self.audio_tokenizer = StreamingMimiModel._from_config(audio_tokenizer_config, dtype=self.dtype)

        self.input_adaptor = EmbeddingAdaptor(
            input_size=int(input_adaptor_config.input_size),
            output_size=int(input_adaptor_config.output_size),
            output_time_scale=float(input_adaptor_config.output_time_scale),
            num_layers=int(input_adaptor_config.num_layers),
            hidden_size=input_adaptor_config.hidden_size,
            decoder_config=input_adaptor_config.decoder_config,
            use_post_norm=bool(input_adaptor_config.use_post_norm),
            norm_eps=float(input_adaptor_config.norm_eps),
            dtype=self.dtype,
        )
        self.output_adaptor = EmbeddingAdaptor(
            input_size=int(output_adaptor_config.input_size),
            output_size=int(output_adaptor_config.output_size),
            output_time_scale=float(output_adaptor_config.output_time_scale),
            num_layers=int(output_adaptor_config.num_layers),
            hidden_size=output_adaptor_config.hidden_size,
            decoder_config=output_adaptor_config.decoder_config,
            use_post_norm=bool(output_adaptor_config.use_post_norm),
            norm_eps=float(output_adaptor_config.norm_eps),
            dtype=self.dtype,
        )

        self.code_predictor = RaonCodePredictor(vllm_config=self.vllm_config, config=code_predictor_config)

        self.make_empty_intermediate_tensors = self.text_model.make_empty_intermediate_tensors

        # Preprocess hook, speaker encoder, tokenizer special tokens.
        self.set_custom_preprocess(self.audio_preprocess)
        self._register_thinker_hook()
        self._init_speaker()
        self._resolve_tokenizer_ids()

    @cached_property
    def sampler(self) -> Sampler:
        return Sampler()

    def _get_audio_decode_state(self, req_id: str | None) -> AudioDecodeState | None:
        if req_id is None:
            return None
        state = self._audio_decode_state.get(req_id)
        if state is None:
            state = AudioDecodeState()
            self._audio_decode_state[req_id] = state
        return state

    def preprocess_begin(self) -> None:
        self._deferred_preprocess.clear()

    def preprocess_end(self) -> None:
        """Flush deferred ``get_audio_output_embeds`` calls into returned tensors."""
        buf = self._deferred_preprocess
        if not buf:
            return

        try:
            for entry in buf:
                codes_tensor = entry.full_codes.unsqueeze(0)
                codes_mask = torch.ones(
                    (1, codes_tensor.shape[1]),
                    device=codes_tensor.device,
                    dtype=torch.bool,
                )
                audio_embeds, _ = self.get_audio_output_embeds(codes_tensor, codes_mask)
                replacement = audio_embeds[0].to(
                    device=entry.input_embeds.device,
                    dtype=entry.input_embeds.dtype,
                )
                entry.input_embeds[entry.audio_positions] = replacement

                step_chunk = entry.full_codes.detach().to(torch.long).to("cpu").contiguous()
                if entry.req_state is not None:
                    entry.req_state.is_generating_audio = True
                    entry.req_state.audio_step_index += int(step_chunk.shape[0])

                existing_total_rows = coerce_optional_int(
                    unwrap_singleton_list(entry.info_dict.get("codec_total_rows"))
                )
                if existing_total_rows is None:
                    existing_total_rows = 0
                total_rows = int(existing_total_rows) + int(step_chunk.shape[0])

                entry.update_dict["codec_codes"] = None
                entry.update_dict["codec_codes_chunk"] = step_chunk
                entry.update_dict["codec_total_rows"] = total_rows
                entry.update_dict["codec_seq"] = max(0, total_rows - 1)
        finally:
            buf.clear()

    def audio_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        update_dict: dict[str, Any] = {}

        if input_embeds is None and input_ids is not None:
            input_embeds = self.embed_input_ids(input_ids)

        if input_ids is None or input_ids.numel() == 0:
            return input_ids, input_embeds, update_dict

        # Insert speaker embedding at <tts_pad> (speaker_token_id) position.
        if self.speaker_token_id is not None:
            speaker_positions = torch.nonzero(input_ids == self.speaker_token_id, as_tuple=False).flatten()
            if speaker_positions.numel() > 0:
                speaker_embeds = info_dict.get("speaker_embeds")
                while isinstance(speaker_embeds, list) and len(speaker_embeds) == 1:
                    speaker_embeds = speaker_embeds[0]
                if not isinstance(speaker_embeds, torch.Tensor):
                    # ICL / voice-cache path: prefer pre-computed embedding.
                    cached_spk = info_dict.get("cached_spk_embedding")
                    cached_spk = unwrap_singleton_list(cached_spk)
                    if isinstance(cached_spk, torch.Tensor):
                        # Validate shape before using; fall back to audio path on mismatch.
                        flat = cached_spk
                        if flat.ndim == 3:
                            flat = flat[:, 0, :]
                        if flat.ndim == 2:
                            flat = flat[0]
                        if flat.ndim == 1 and int(flat.shape[0]) == int(input_embeds.shape[-1]):
                            speaker_embeds = cached_spk
                            update_dict["speaker_embeds"] = speaker_embeds
                        else:
                            logger.warning(
                                "cached_spk_embedding shape %s incompatible with hidden=%d; falling back",
                                tuple(cached_spk.shape),
                                int(input_embeds.shape[-1]),
                            )

                if not isinstance(speaker_embeds, torch.Tensor):
                    speaker_ref_audio = normalize_speaker_ref_audio(
                        info_dict.get("speaker_ref_audio", info_dict.get("ref_audio"))
                    )
                    if speaker_ref_audio is not None and self.speaker_encoder is not None:
                        loaded = load_speaker_ref_audio(speaker_ref_audio)
                        if loaded is not None:
                            audio, sr = loaded
                            if audio.ndim > 1 and audio.shape[0] > 1:
                                audio = audio.mean(dim=0, keepdim=True)
                            speaker_audio = audio.to(dtype=torch.float32).transpose(0, 1).contiguous().view(1, -1)
                            speaker_lengths = torch.tensor([speaker_audio.shape[1]], dtype=torch.long)
                            speaker_embeds = compute_speaker_embeds(
                                self.speaker_encoder,
                                audio=speaker_audio,
                                audio_lengths=speaker_lengths,
                                sampling_rate=int(sr),
                                model_sampling_rate=self.sampling_rate,
                            )
                            update_dict["speaker_embeds"] = speaker_embeds
                            update_dict["speaker_ref_audio"] = None
                        else:
                            logger.warning(
                                "Skipping speaker conditioning; unable to load ref audio: %s",
                                speaker_ref_audio,
                            )

                if isinstance(speaker_embeds, torch.Tensor):
                    embeds = speaker_embeds
                    if embeds.ndim == 3:
                        embeds = embeds[:, 0, :]
                    if embeds.ndim == 2:
                        embeds = embeds[0]
                    if embeds.ndim != 1 or int(embeds.shape[0]) != int(input_embeds.shape[-1]):
                        raise AssertionError(
                            "speaker_embeds must resolve to a single vector matching hidden size: "
                            f"got shape={tuple(speaker_embeds.shape)}, hidden={int(input_embeds.shape[-1])}."
                        )

                    replacement = embeds.to(device=input_embeds.device, dtype=input_embeds.dtype)
                    input_embeds = input_embeds.clone()
                    input_embeds[speaker_positions] = replacement.expand(int(speaker_positions.numel()), -1)

        # Audio-code feedback only applies to decode ticks (1 token/request).
        if int(input_ids.shape[0]) != 1:
            return input_ids, input_embeds, update_dict

        # Audio output tokens are AUDIO_OUTPUT_PLACEHOLDER; codes live on per-request state.
        audio_positions = torch.nonzero(
            input_ids == self.audio_output_token_id,
            as_tuple=False,
        ).flatten()
        if audio_positions.numel() == 0:
            return input_ids, input_embeds, update_dict

        req_id = (
            normalize_runtime_request_id(info_dict.get("global_request_id", info_dict.get("_omni_req_id")))
            or "__unknown__"
        )
        req_state = self._get_audio_decode_state(None if req_id == "__unknown__" else req_id)

        # First audio step: keep learned trigger embedding only when no
        # pending semantic codes are available yet (bootstrap step).
        is_first_audio_step = req_state is None or int(req_state.audio_step_index) == 0
        if is_first_audio_step:
            forced_first_token = bool(torch.all(input_ids[audio_positions] == int(self.audio_output_token_id)))
            pending_codes = req_state.pending_audio_codes if req_state is not None else None
            if forced_first_token and pending_codes is None:
                return input_ids, input_embeds, update_dict

        # Read pre-generated audio codes from AudioDecodeState (produced in compute_logits).
        if req_state is None or req_state.pending_audio_codes is None:
            logger.warning(
                "Missing pending_audio_codes for req_id=%s; leaving embed unchanged",
                req_id,
            )
            return input_ids, input_embeds, update_dict
        full_codes = req_state.pending_audio_codes
        req_state.pending_audio_codes = None

        input_embeds = input_embeds.clone()

        self._deferred_preprocess.append(
            _DeferredPreprocessEntry(
                input_embeds=input_embeds,
                audio_positions=audio_positions,
                full_codes=full_codes,
                update_dict=update_dict,
                req_state=req_state,
                info_dict=info_dict,
            )
        )

        return input_ids, input_embeds, update_dict

    def _build_audio_encoder(
        self,
        cfg: Qwen3OmniMoeAudioEncoderConfig | None,
    ) -> nn.Module:
        """Build Qwen3 Omni MoE audio encoder (Whisper-style AuT)."""
        if cfg is None:
            raise ValueError("Raon requires `audio_encoder_config` to build the audio encoder.")
        return Qwen3OmniAuTWrapper.from_config(config=cfg, dtype=self.dtype)

    def _init_speaker(self) -> None:
        """Initialize speaker token ID and speaker encoder."""
        self.speaker_token_id: int | None = None
        speaker_token_id = self.config.speaker_token_id
        if isinstance(speaker_token_id, int):
            self.speaker_token_id = int(speaker_token_id)
        elif isinstance(speaker_token_id, (list, tuple)):
            self.speaker_token_id = int(speaker_token_id[0]) if speaker_token_id else None

        self.speaker_encoder: PretrainedSpeakerEncoder | None = None
        self.is_pretrained_speaker_encoder = False
        speaker_cfg = coerce_speaker_encoder_config(self.config.speaker_encoder_config)
        if isinstance(speaker_cfg, SpeakerEncoderConfig):
            self.speaker_encoder = build_speaker_encoder(speaker_cfg, dtype=self.dtype)
            self.is_pretrained_speaker_encoder = isinstance(self.speaker_encoder, PretrainedSpeakerEncoder)
            if self.is_pretrained_speaker_encoder and isinstance(self.speaker_encoder, PretrainedSpeakerEncoder):
                try:
                    # Warm artifacts eagerly to avoid stalling on first TTS request
                    self.speaker_encoder.warm_backend_artifacts()
                    logger.info(
                        "Warmed pretrained speaker artifacts: encoder_type=%s model_id=%s",
                        self.speaker_encoder.encoder_type,
                        self.speaker_encoder.pretrained_model_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to warm pretrained speaker artifacts; "
                        "speaker-conditioned TTS will use fully lazy init: %s",
                        exc,
                    )

    def _register_thinker_hook(self) -> None:
        """Register forward hook to capture thinker hidden states."""

        def _hook(module, input, output):
            # Reconstruct full hidden state (vLLM splits residual stream)
            if isinstance(output, tuple) and len(output) >= 2:
                logger.debug(
                    "[ThinkerHook] output types=%s shapes=%s",
                    [type(o).__name__ for o in output[:3]],
                    [tuple(o.shape) if hasattr(o, 'shape') else 'N/A' for o in output[:3]],
                )
                thinker_hidden = output[0] + output[1]  # mlp + residual
            elif isinstance(output, tuple):
                thinker_hidden = output[0]
            else:
                thinker_hidden = output
            if isinstance(thinker_hidden, torch.Tensor):
                self._thinker_hidden_queue.append(thinker_hidden)
                if len(self._thinker_hidden_queue) > self._MAX_HIDDEN_QUEUE_DEPTH:
                    self._thinker_hidden_queue.popleft()

        self.text_model.layers[self.accept_hidden_layer].register_forward_hook(_hook)

    def _resolve_tokenizer_ids(self) -> None:
        """Load tokenizer and resolve/override token IDs."""
        eos_token_id = getattr(self.config.text_model_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        self.eos_token_id: int | None = int(eos_token_id) if isinstance(eos_token_id, int) else None
        self._audio_only_allowed_text_token_ids: tuple[int, ...] = (
            (self.eos_token_id,) if self.eos_token_id is not None else ()
        )

        self._tokenizer_len: int | None = None
        self.audio_end_token_id: int | None = None
        self._tokenizer = None
        try:
            tok_path = self.vllm_config.model_config.tokenizer or self.vllm_config.model_config.model
            tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                trust_remote_code=self.vllm_config.model_config.trust_remote_code,
                fix_mistral_regex=True,
            )
            align_tokenizer(tokenizer)
            self._tokenizer = tokenizer
            self._tokenizer_len = len(tokenizer)
            tok_eos = getattr(tokenizer, "eos_token_id", None)
            if isinstance(tok_eos, (list, tuple)):
                tok_eos = tok_eos[0] if tok_eos else None
            if isinstance(tok_eos, int):
                self.eos_token_id = int(tok_eos)

            allowed: list[int] = []
            if self.eos_token_id is not None:
                allowed.append(self.eos_token_id)
            for tok in (AUDIO_START_TOKEN, AUDIO_END_TOKEN):
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if len(ids) == 1 and isinstance(ids[0], int):
                    if tok == AUDIO_END_TOKEN:
                        self.audio_end_token_id = int(ids[0])
                    allowed.append(int(ids[0]))
            allowed.append(int(self.audio_output_token_id))

            # Override audio_input_token_id from tokenizer if stale
            audio_input_ids = tokenizer.encode(AUDIO_INPUT_PAD_TOKEN, add_special_tokens=False)
            if len(audio_input_ids) == 1 and isinstance(audio_input_ids[0], int):
                tokenizer_audio_input_token_id = int(audio_input_ids[0])
                if self.audio_input_token_id != tokenizer_audio_input_token_id:
                    logger.warning(
                        "Overriding audio_input_token_id from %s to tokenizer id %s",
                        self.audio_input_token_id,
                        tokenizer_audio_input_token_id,
                    )
                    self.audio_input_token_id = tokenizer_audio_input_token_id

            # Override speaker_token_id from tokenizer if available
            if self.speaker_encoder is not None:
                try:
                    tokenizer_speaker_token_id = resolve_speaker_token_id(
                        tokenizer,
                        expected_speaker_token_id=self.speaker_token_id,
                    )
                    if self.speaker_token_id != tokenizer_speaker_token_id:
                        logger.warning(
                            "Overriding speaker_token_id from %s to tokenizer id %s",
                            self.speaker_token_id,
                            tokenizer_speaker_token_id,
                        )
                        self.speaker_token_id = tokenizer_speaker_token_id
                except Exception as exc:
                    logger.warning(
                        "Speaker token lookup failed; speaker-conditioned TTS may be disabled: %s",
                        exc,
                    )

            if allowed:
                seen: set[int] = set()
                unique = [tid for tid in allowed if not (tid in seen or seen.add(tid))]
                self._audio_only_allowed_text_token_ids = tuple(unique)
        except Exception as exc:
            logger.warning("Tokenizer length lookup failed; detok bound checks disabled: %s", exc)

    @torch.inference_mode()
    def get_audio_output_embeds_from_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference audio → Mimi codes → output-adaptor embeddings (thinker ICL)."""
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[:, None]
        elif audio.ndim != 3:
            raise ValueError(f"audio must be 1D/2D/3D tensor, got shape={tuple(audio.shape)}.")

        max_samples = ENV.max_audio_duration_s * self.sampling_rate
        if audio.shape[-1] > max_samples:
            raise ValueError(
                f"Audio input too long: {audio.shape[-1]} samples "
                f"({audio.shape[-1] / self.sampling_rate:.1f}s), "
                f"max allowed {ENV.max_audio_duration_s}s"
            )

        target_device = module_device(self.audio_tokenizer)
        target_dtype = module_dtype(self.audio_tokenizer)
        audio = audio.to(device=target_device, dtype=target_dtype)

        if audio_lengths is None:
            audio_lengths = torch.full(
                (int(audio.shape[0]),), int(audio.shape[-1]),
                dtype=torch.long, device=audio.device,
            )
        else:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long).reshape(-1)

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=int(sampling_rate), new_freq=int(self.sampling_rate),
            )
            audio_lengths = (audio_lengths.float() * float(self.sampling_rate) / float(sampling_rate)).long()

        audio_lengths = audio_lengths.clamp(min=1, max=int(audio.shape[-1]))
        indices = torch.arange(audio.shape[-1], device=audio.device)
        audio_mask = (indices[None] < audio_lengths[:, None]).long()

        outputs = self.audio_tokenizer.encode(
            audio, padding_mask=audio_mask,
            num_quantizers=int(self.num_code_groups),
            return_dict=True,
        )
        if getattr(outputs, "audio_codes", None) is None:
            raise RuntimeError("audio_tokenizer.encode returned no audio_codes for ICL output embeddings.")

        audio_codes = outputs.audio_codes.view(outputs.audio_codes.shape[-3:]).transpose(1, 2)
        padded_audio_mask = F.pad(
            audio_mask,
            (0, int(audio_codes.shape[1]) * int(self.samples_per_frame) - int(audio_mask.shape[1])),
        )
        audio_codes_mask = padded_audio_mask.view(-1, audio_codes.shape[1], self.samples_per_frame).any(dim=-1)
        return self.get_audio_output_embeds(audio_codes, audio_codes_mask)

    def _get_silence_codes(self) -> torch.Tensor:
        """Encode actual silence through Mimi and cache the resulting codes [N, G]."""
        if self._cached_silence_codes is None:
            n = self._N_ICL_SILENCE_FRAMES
            dur = int(n * self.samples_per_frame)
            dev = next(self.audio_tokenizer.parameters()).device
            wav = torch.zeros(1, 1, dur, device=dev, dtype=torch.bfloat16)
            mask = torch.ones(1, dur, device=dev, dtype=torch.long)
            with torch.inference_mode():
                out = self.audio_tokenizer.encode(wav, mask, num_quantizers=int(self.num_code_groups), return_dict=True)
            # out.audio_codes: [1, G, T] → transpose to [1, T, G] → take [0] → [T, G]
            self._cached_silence_codes = out.audio_codes.view(out.audio_codes.shape[-3:]).transpose(1, 2)[0].to(torch.long).cpu().contiguous()
        return self._cached_silence_codes

    def build_request_payload(
        self,
        req_id: str,
        req_info: dict[str, Any],
        is_finished: bool,
    ) -> dict[str, object]:
        """Stage bridge codec payload: per-step chunk while running; full codes when the request finishes."""
        codec_payload: torch.Tensor | None = None

        if is_finished:
            codec_full = req_info.get("codec_codes")
            if isinstance(codec_full, torch.Tensor) and codec_full.numel() > 0:
                codec_payload = collapse_exact_repeated_codec_snapshot(codec_full)
                req_info["codec_codes"] = codec_payload
                req_info["codec_total_rows"] = int(codec_payload.shape[0])
                req_info["codec_seq"] = max(0, int(codec_payload.shape[0]) - 1)
            # Drop any leftover chunk state for cleanliness.
            req_info.pop("codec_codes_chunk", None)
        else:
            codec_chunk = req_info.pop("codec_codes_chunk", None)
            if isinstance(codec_chunk, torch.Tensor) and codec_chunk.numel() > 0:
                codec_payload = codec_chunk

        result: dict[str, object] = {}
        if isinstance(codec_payload, torch.Tensor) and codec_payload.numel() > 0:
            result["codec_codes"] = codec_payload.detach().to("cpu").contiguous()
        for meta_key in ("global_request_id", "source_text"):
            meta_value = req_info.get(meta_key)
            if meta_value is not None:
                result[meta_key] = meta_value
        return result

    @torch.inference_mode()
    def get_audio_input_embeds(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[:, None]
        elif audio.ndim != 3:
            raise ValueError(f"audio must be 1D/2D/3D tensor, got shape={tuple(audio.shape)}.")

        max_samples = ENV.max_audio_duration_s * self.sampling_rate
        if audio.shape[-1] > max_samples:
            raise ValueError(
                f"Audio input too long: {audio.shape[-1]} samples "
                f"({audio.shape[-1] / self.sampling_rate:.1f}s), "
                f"max allowed {ENV.max_audio_duration_s}s"
            )

        target_device = module_device(self.audio_encoder)
        target_dtype = module_dtype(self.audio_encoder)
        audio = audio.to(device=target_device, dtype=target_dtype)

        encoder_sampling_rate = int(getattr(self.audio_encoder.config, "sampling_rate", self.sampling_rate))
        if sampling_rate is not None and sampling_rate != encoder_sampling_rate:
            audio = torchaudio.functional.resample(
                waveform=audio.float(),
                orig_freq=sampling_rate,
                new_freq=encoder_sampling_rate,
            )
            audio = audio.to(dtype=target_dtype)

        encoder_outputs = self.audio_encoder(audio, use_streaming=False)
        audio_embeds = getattr(encoder_outputs, "embeds", None)
        if not isinstance(audio_embeds, torch.Tensor):
            raise RuntimeError("audio_encoder returned no audio embeddings.")

        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long).reshape(-1)
            if int(audio_lengths.shape[0]) != int(audio.shape[0]):
                raise ValueError(
                    f"audio_lengths batch mismatch: got {int(audio_lengths.shape[0])}, expected {int(audio.shape[0])}."
                )
            audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

            indices = torch.arange(audio.shape[-1], device=audio.device)
            audio_embeds_mask = (indices[None] < audio_lengths[:, None]).long()
            target_audio_samples = audio_embeds.shape[1] * self.samples_per_frame
            if target_audio_samples < audio_embeds_mask.shape[1]:
                raise ValueError(
                    "audio_encoder produced fewer frames than expected from input length: "
                    f"target_samples={target_audio_samples}, "
                    f"input_samples={audio_embeds_mask.shape[1]}."
                )
            padded_audio_mask = F.pad(
                audio_embeds_mask,
                (0, target_audio_samples - audio_embeds_mask.shape[1]),
            )
            audio_embeds_mask = padded_audio_mask.view(-1, audio_embeds.shape[1], self.samples_per_frame).any(dim=-1)
        else:
            audio_embeds_mask = torch.ones(
                audio_embeds.shape[:2],
                dtype=torch.bool,
                device=audio_embeds.device,
            )

        adaptor_outputs = self.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        if adaptor_outputs.mask is None:
            adaptor_outputs.mask = torch.ones(
                adaptor_outputs.outputs_embeds.shape[:2],
                dtype=torch.bool,
                device=adaptor_outputs.outputs_embeds.device,
            )
        return adaptor_outputs.outputs_embeds, adaptor_outputs.mask

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_waveforms = kwargs.pop("audio_waveforms", None)
        audio_lengths = kwargs.pop("audio_lengths", None)
        if audio_waveforms is None:
            return []

        audio_waveforms, audio_lengths = normalize_audio_waveforms_and_lengths(
            audio_waveforms,
            audio_lengths,
        )
        target_device = module_device(self.audio_encoder)
        audio_waveforms = audio_waveforms.to(device=target_device)
        audio_lengths = audio_lengths.to(device=target_device)

        # Per-audio placeholder routing: check if any audio should use
        # the output-embed path (ICL prefill via Mimi codec).
        audio_placeholder_token_ids = kwargs.pop("audio_placeholder_token_ids", None)
        if audio_placeholder_token_ids is not None:
            if isinstance(audio_placeholder_token_ids, list):
                audio_placeholder_token_ids = torch.as_tensor(audio_placeholder_token_ids, dtype=torch.long)
            elif isinstance(audio_placeholder_token_ids, torch.Tensor):
                audio_placeholder_token_ids = audio_placeholder_token_ids.to(dtype=torch.long).reshape(-1)
        if audio_placeholder_token_ids is None:
            audio_placeholder_token_ids = torch.full(
                (int(audio_waveforms.shape[0]),), int(self.audio_input_token_id), dtype=torch.long,
            )

        per_audio: list[torch.Tensor] = []
        for idx in range(int(audio_waveforms.shape[0])):
            item_audio = audio_waveforms[idx: idx + 1]
            item_lengths = audio_lengths[idx: idx + 1]
            placeholder_tid = int(audio_placeholder_token_ids[idx].item())

            if placeholder_tid == int(self.audio_output_token_id):
                # ICL path: check for cached Mimi codec codes (skip expensive encode)
                cached_codes_raw = kwargs.get("cached_ref_codec_codes")
                cached_mask_raw = kwargs.get("cached_ref_codec_codes_mask")

                # Unwrap singleton lists from additional_information
                if isinstance(cached_codes_raw, list) and len(cached_codes_raw) > 0:
                    cached_codes_raw = cached_codes_raw[0]
                if isinstance(cached_mask_raw, list) and len(cached_mask_raw) > 0:
                    cached_mask_raw = cached_mask_raw[0]

                if isinstance(cached_codes_raw, torch.Tensor) and cached_codes_raw.numel() > 0:
                    _cached_codes = cached_codes_raw.to(device=target_device, dtype=torch.long)
                    if _cached_codes.ndim == 2:
                        _cached_codes = _cached_codes.unsqueeze(0)  # [T, G] -> [1, T, G]
                    if isinstance(cached_mask_raw, torch.Tensor):
                        _cached_mask = cached_mask_raw.to(device=target_device)
                        if _cached_mask.ndim == 1:
                            _cached_mask = _cached_mask.unsqueeze(0)
                    else:
                        _cached_mask = torch.ones(
                            _cached_codes.shape[:2], dtype=torch.bool, device=target_device,
                        )
                    item_embeds, item_mask = self.get_audio_output_embeds(_cached_codes, _cached_mask)
                else:
                    item_embeds, item_mask = self.get_audio_output_embeds_from_audio(
                        audio=item_audio, audio_lengths=item_lengths,
                    )
            else:
                # Standard path: audio encoder → input adaptor
                item_embeds, item_mask = self.get_audio_input_embeds(
                    audio=item_audio, audio_lengths=item_lengths,
                )
            per_audio.append(item_embeds[0][item_mask[0]])
        return tuple(per_audio)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.text_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None:
            return inputs_embeds
        if isinstance(multimodal_embeddings, (list, tuple)) and len(multimodal_embeddings) == 0:
            return inputs_embeds

        audio_embeddings = flatten_audio_embeddings(
            multimodal_embeddings,
            hidden_size=inputs_embeds.shape[-1],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        # Placeholder token id used to scatter multimodal embeddings (default: audio_input).
        scatter_token_id = self.audio_input_token_id
        if is_multimodal is not None:
            mm_token_ids = input_ids[is_multimodal]
            unique_ids = mm_token_ids.unique()
            if unique_ids.numel() == 1:
                scatter_token_id = int(unique_ids.item())
            elif unique_ids.numel() > 1:
                # Mixed placeholder types in batch (e.g. audio_input + audio_output
                # from concurrent STT and TTS requests).  Scatter each type
                # separately so the single-token-id guard passes.
                for tid in unique_ids.tolist():
                    type_mm = is_multimodal & (input_ids == tid)
                    type_emb = audio_embeddings[mm_token_ids == tid]
                    inputs_embeds = scatter_audio_input_embeddings(
                        inputs_embeds=inputs_embeds,
                        input_ids=input_ids,
                        audio_input_embeddings=type_emb,
                        audio_input_token_id=tid,
                        is_multimodal=type_mm,
                    )
                return inputs_embeds

        return scatter_audio_input_embeddings(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            audio_input_embeddings=audio_embeddings,
            audio_input_token_id=scatter_token_id,
            is_multimodal=is_multimodal,
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.text_model

    def _get_text_logits_hidden(
        self,
        hidden_states: torch.Tensor,
        thinker_hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        # Split checkpoints expose thinker output after text_model.norm.
        del thinker_hidden
        return hidden_states

    def _collect_live_request_ids(
        self,
        runtime_additional_information: list[dict[str, Any]] | None,
    ) -> list[str]:
        if not isinstance(runtime_additional_information, list):
            return []
        live_req_ids: list[str] = []
        for info in runtime_additional_information:
            if not isinstance(info, dict):
                continue
            req_id = normalize_runtime_request_id(info.get("global_request_id", info.get("_omni_req_id")))
            if req_id is not None:
                live_req_ids.append(req_id)
        return live_req_ids

    def _enqueue_runtime_info(
        self,
        runtime_additional_information: list[dict[str, Any]] | None,
    ) -> None:
        live_req_ids = self._collect_live_request_ids(runtime_additional_information)
        has_stale_split_state = (
            len(self._thinker_hidden_queue) > 0
            or len(self._talker_hidden_for_logits_queue) > 0
            or len(self._talker_hidden_for_postprocess_queue) > 0
        )
        if live_req_ids and len(self._runtime_info_queue) == 0 and has_stale_split_state:
            logger.warning(
                "Clearing stale split queues before "
                "req_ids=%s thinker=%d talker_logits=%d talker_post=%d",
                ",".join(live_req_ids),
                len(self._thinker_hidden_queue),
                len(self._talker_hidden_for_logits_queue),
                len(self._talker_hidden_for_postprocess_queue),
            )
            self._thinker_hidden_queue.clear()
            self._talker_hidden_for_logits_queue.clear()
            self._talker_hidden_for_postprocess_queue.clear()
            self._runtime_info_queue.clear()

        self._runtime_info_queue.append(runtime_additional_information or [])
        if len(self._runtime_info_queue) > self._MAX_HIDDEN_QUEUE_DEPTH:
            self._runtime_info_queue.popleft()

    def _pop_thinker_hidden_for_forward(self) -> torch.Tensor:
        thinker_hidden = self._thinker_hidden_queue.popleft() if len(self._thinker_hidden_queue) > 0 else None
        if thinker_hidden is None:
            raise RuntimeError("Missing thinker hidden state for split talker forward.")
        return thinker_hidden

    @staticmethod
    def _gather_last_hidden_per_request(
        hidden: torch.Tensor,
        runtime_info: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        """Select the last token per request from *hidden* [total_tokens, D].

        Uses ``_num_tokens`` injected by the model runner to compute the
        cumulative last-token indices — the same positions the model runner
        selects for ``compute_logits``.
        """
        if not isinstance(runtime_info, list) or not runtime_info:
            return hidden
        num_reqs = len(runtime_info)
        total = hidden.shape[0]
        if total <= num_reqs:
            return hidden
        # Build last-token indices from per-request token counts.
        cum = 0
        indices: list[int] = []
        for info in runtime_info:
            n = int(info.get("_num_tokens", 1)) if isinstance(info, dict) else 1
            cum += n
            indices.append(cum - 1)
        if not indices or indices[-1] >= total:
            logger.warning("_gather_last_hidden: index %s >= total %d, skipping", indices[-1:], total)
            return hidden
        return hidden[torch.tensor(indices, device=hidden.device, dtype=torch.long)]

    def _enqueue_talker_hidden_for_runtime(self, talker_hidden_states: torch.Tensor | object) -> None:
        if not isinstance(talker_hidden_states, torch.Tensor):
            return
        self._talker_hidden_for_logits_queue.append(talker_hidden_states)
        self._talker_hidden_for_postprocess_queue.append(talker_hidden_states)
        while len(self._talker_hidden_for_logits_queue) > self._MAX_HIDDEN_QUEUE_DEPTH:
            self._talker_hidden_for_logits_queue.popleft()
        while len(self._talker_hidden_for_postprocess_queue) > self._MAX_HIDDEN_QUEUE_DEPTH:
            self._talker_hidden_for_postprocess_queue.popleft()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        self._enqueue_runtime_info(runtime_additional_information)

        text_hidden_states = self.text_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if not isinstance(text_hidden_states, torch.Tensor):
            return text_hidden_states

        thinker_hidden = self._pop_thinker_hidden_for_forward()
        talker_input = self.thinker_to_talker_proj(thinker_hidden.to(dtype=self.dtype))
        talker_hidden_states = self.talker(
            input_ids=None,
            positions=positions,
            inputs_embeds=talker_input,
        )
        # Align talker hidden states with compute_logits: forward() produces
        # [total_tokens, hidden] but compute_logits receives only the last
        # token per request after the model-runner's gathering step.
        # Select the same last-token positions here so the queue shape matches.
        talker_hidden_states = self._gather_last_hidden_per_request(
            talker_hidden_states, runtime_additional_information,
        )
        self._enqueue_talker_hidden_for_runtime(talker_hidden_states)
        return text_hidden_states

    def make_omni_output(self, model_outputs: torch.Tensor | IntermediateTensors | OmniOutput, **kwargs: Any):
        if isinstance(model_outputs, (OmniOutput, IntermediateTensors)):
            return model_outputs

        runtime_info = kwargs.get("runtime_additional_information", [])
        per_req_full: list[torch.Tensor | None] = []
        per_req_chunk: list[torch.Tensor | None] = []
        if isinstance(runtime_info, list):
            # Batch-aligned with runtime_additional_information; dedupe duplicate
            # req_ids by keeping the entry with the largest seq (see codec_seq / rows).

            def _parse_codec_slot(info: dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None, int]:
                full = info.get("codec_codes")
                full = full if isinstance(full, torch.Tensor) and full.numel() > 0 else None
                chunk = info.get("codec_codes_chunk")
                chunk = chunk if isinstance(chunk, torch.Tensor) and chunk.numel() > 0 else None
                req_id = normalize_runtime_request_id(info.get("global_request_id", info.get("_omni_req_id")))
                seq = coerce_optional_int(info.get("codec_seq"))
                if seq is None:
                    tr = coerce_optional_int(info.get("codec_total_rows"))
                    if tr is not None:
                        seq = max(0, int(tr) - 1)
                if seq is None:
                    if full is not None:
                        seq = max(0, int(full.shape[0]) - 1)
                    elif chunk is not None:
                        seq = max(0, int(chunk.shape[0]) - 1)
                    else:
                        seq = -1
                return full, chunk, req_id, seq

            parsed: list[tuple[torch.Tensor | None, torch.Tensor | None, str | None, int] | None] = []
            for info in runtime_info:
                parsed.append(_parse_codec_slot(info) if isinstance(info, dict) else None)

            selected_by_req: dict[str, tuple[int, torch.Tensor | None, torch.Tensor | None]] = {}
            total_entries = 0
            fallback_entries = 0
            for slot in parsed:
                if slot is None:
                    continue
                full, chunk, req_id, seq = slot
                if full is None and chunk is None:
                    continue
                total_entries += 1
                if req_id is None:
                    fallback_entries += 1
                    continue
                prev = selected_by_req.get(req_id)
                if prev is None or seq >= prev[0]:
                    selected_by_req[req_id] = (seq, full, chunk)

            for slot in parsed:
                if slot is None:
                    per_req_full.append(None)
                    per_req_chunk.append(None)
                    continue
                full, chunk, req_id, _ = slot
                if req_id is not None:
                    sel = selected_by_req.get(req_id)
                    if sel is not None:
                        _, full, chunk = sel
                per_req_full.append(full)
                per_req_chunk.append(chunk)

            selected_entries = len(selected_by_req) + fallback_entries
            if total_entries > selected_entries:
                logger.warning(
                    "Deduped runtime codec payload entries while preserving batch alignment: "
                    "total=%d selected=%d",
                    total_entries,
                    selected_entries,
                )

        multimodal_outputs: dict[str, Any] = {}
        if any(item is not None for item in per_req_full):
            multimodal_outputs["codec_codes"] = per_req_full
        if any(item is not None for item in per_req_chunk):
            multimodal_outputs["codec_codes_chunk"] = per_req_chunk

        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=multimodal_outputs)

    def _normalize_output_mode(self, req_info: dict[str, Any] | None) -> str:
        if not isinstance(req_info, dict):
            return "text_and_audio"
        value = req_info.get("output_mode", "text_and_audio")
        if isinstance(value, list):
            value = value[0] if value else "text_and_audio"
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else "text_and_audio"
        if not isinstance(value, str):
            return "text_and_audio"
        value = value.lower().strip()
        if value in {"text_only", "audio_only", "text_and_audio"}:
            return value
        return "text_and_audio"

    def _mask_audio_logits_for_text_mode(self, logits: torch.Tensor, row_idx: int) -> None:
        audio_start_idx = int(self.audio_output_token_id)
        if 0 <= audio_start_idx < int(logits.shape[-1]):
            logits[row_idx, audio_start_idx:] = float("-inf")

    @staticmethod
    def _apply_audio_sampling_params(logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature / top-k / top-p warping to audio_lm_head logits."""
        temperature = ENV.tts_temperature
        top_k = ENV.tts_top_k
        top_p = ENV.tts_top_p

        scores = logits.float()
        if temperature > 0 and temperature != 1.0:
            scores = scores / temperature
        if top_k > 0:
            kth, _ = torch.topk(scores, min(top_k, scores.shape[-1]))
            scores = scores.masked_fill(scores < kth[..., -1:], float("-inf"))
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scores, descending=False)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cum_probs <= (1.0 - top_p)
            remove[..., -1:] = False
            remove_orig = remove.scatter(-1, sorted_idx, remove)
            scores = scores.masked_fill(remove_orig, float("-inf"))
        return scores

    def _generate_audio_for_row(
        self,
        *,
        logits: torch.Tensor,
        row_idx: int,
        req_info: dict[str, Any] | None,
        req_runtime_id: str | None,
        req_state: AudioDecodeState | None,
        force_audio_first_token: bool,
        is_sampled_row: bool,
        row_output_ids: list[int],
        audio_hidden_states: torch.Tensor | None,
    ) -> None:
        forced_audio_bootstrap = False
        if force_audio_first_token and is_sampled_row:
            if req_state is not None:
                forced_audio_bootstrap = not bool(req_state.forced_audio_bootstrap_done)
            else:
                # Fallback for missing request-id metadata.
                forced_audio_bootstrap = not row_output_ids

        if forced_audio_bootstrap:
            if req_state is not None:
                req_state.forced_audio_bootstrap_done = True
            logits[row_idx, :] = float("-inf")
            if 0 <= int(self.audio_output_token_id) < int(logits.shape[-1]):
                logits[row_idx, int(self.audio_output_token_id)] = 0.0
            else:
                logger.warning(
                    "Invalid "
                    "audio_output_token_id=%s for vocab=%s; "
                    "skipping forced audio bootstrap",
                    self.audio_output_token_id,
                    int(logits.shape[-1]),
                )
            return

        if not is_sampled_row:
            return

        audio_logits_row = (
            self.audio_lm_head(audio_hidden_states[row_idx : row_idx + 1]).squeeze(0)
            if audio_hidden_states is not None
            else None
        )
        if audio_logits_row is None:
            # Defensive: force audio_output_token_id to prevent text/EOS token selection
            logits[row_idx, :] = float("-inf")
            if 0 <= int(self.audio_output_token_id) < int(logits.shape[-1]):
                logits[row_idx, int(self.audio_output_token_id)] = 0.0
            return

        # Determine if we're in the silence window
        in_silence_window = (
            req_state is not None
            and req_state.continuation_silence_frames > 0
            and req_state.audio_step_index < req_state.continuation_silence_frames
        )

        # Suppress AUDIO_END during silence window + 1 grace step
        _eos_suppress_until = (
            req_state.continuation_silence_frames + 2
            if req_state is not None and req_state.continuation_silence_frames > 0
            else 0
        )
        if req_state is not None and req_state.audio_step_index < _eos_suppress_until:
            audio_logits_row = audio_logits_row.clone()
            audio_logits_row[self.codebook_size] = float("-inf")

        if ENV.tts_temperature != 1.0 or ENV.tts_top_k > 0 or ENV.tts_top_p < 1.0:
            audio_logits_row = self._apply_audio_sampling_params(audio_logits_row)
        first_code = int(torch.multinomial(torch.softmax(audio_logits_row.float(), dim=-1), 1).item())

        if in_silence_window:
            silence = self._get_silence_codes()
            step_idx = req_state.audio_step_index
            if step_idx < silence.shape[0]:
                silence_frame = silence[step_idx]  # [G]
            else:
                silence_frame = silence[-1]  # repeat last if needed

            # Build full_codes from silence frame
            row_state = req_state if req_state is not None else self._get_audio_decode_state(req_runtime_id)
            row_state.pending_audio_codes = silence_frame.unsqueeze(0).to(device=logits.device)
            row_state.is_generating_audio = True
            # audio_step_index is incremented by preprocess_end (line 356); do NOT double-increment here

            # Force audio_output_token_id in text logits (same as normal audio generation)
            logits[row_idx, :] = float("-inf")
            if 0 <= int(self.audio_output_token_id) < int(logits.shape[-1]):
                logits[row_idx, int(self.audio_output_token_id)] = 0.0

            # Do NOT record in RAS history (silence frames are artificial)
            return

        first_code = self._ras.maybe_resample(
            req_runtime_id,
            first_code,
            audio_logits_row,
            self.codebook_size,
        )

        if first_code == self.codebook_size:
            # AUDIO_END: force the audio end token.
            logits[row_idx, :] = float("-inf")
            if isinstance(self.audio_end_token_id, int) and 0 <= self.audio_end_token_id < int(logits.shape[-1]):
                logits[row_idx, self.audio_end_token_id] = 0.0
            return

        row_state = req_state if req_state is not None else self._get_audio_decode_state(req_runtime_id)
        first_code_tensor = torch.tensor([[first_code]], device=logits.device, dtype=torch.long)
        audio_hidden_row = audio_hidden_states[row_idx : row_idx + 1] if audio_hidden_states is not None else None

        speaker_embed: torch.Tensor | None = None
        if isinstance(req_info, dict):
            speaker_embed = unwrap_singleton_list(req_info.get("speaker_embeds"))
        speaker_batch: torch.Tensor | None = None
        if isinstance(speaker_embed, torch.Tensor) and self.proj_speaker_code is not None:
            if speaker_embed.ndim == 3:
                speaker_embed = speaker_embed[:, 0, :]
            if speaker_embed.ndim == 2:
                speaker_embed = speaker_embed[0]
            if speaker_embed.ndim == 1:
                speaker_batch = speaker_embed.to(
                    device=logits.device,
                    dtype=self.proj_code.weight.dtype,
                ).view(1, 1, -1)

        if audio_hidden_row is not None:
            hidden_for_code = audio_hidden_row.to(dtype=self.proj_code.weight.dtype)
            if hidden_for_code.ndim == 2:
                hidden_for_code = hidden_for_code.unsqueeze(1)
            full_codes = self.generate_audio_codes(
                input_ids=first_code_tensor,
                inputs_embeds=self.proj_code(hidden_for_code),
                num_code_groups=self.num_code_groups,
                speaker_embeds=speaker_batch,
            ).to(torch.long)
            if row_state is not None:
                row_state.pending_audio_codes = full_codes

            self._ras.record(req_runtime_id, first_code)

        logits[row_idx, :] = float("-inf")
        if 0 <= int(self.audio_output_token_id) < int(logits.shape[-1]):
            logits[row_idx, int(self.audio_output_token_id)] = 0.0

    def _peek_runtime_info(self) -> list[Any]:
        queued_runtime_info = self._runtime_info_queue[0] if len(self._runtime_info_queue) > 0 else []
        return queued_runtime_info if isinstance(queued_runtime_info, list) else []

    def _pop_audio_hidden_for_logits(
        self,
        hidden_states: torch.Tensor,
        queued_runtime_info: list[Any],
    ) -> torch.Tensor | None:
        audio_hidden_states = (
            self._talker_hidden_for_logits_queue.popleft() if len(self._talker_hidden_for_logits_queue) > 0 else None
        )
        if not isinstance(audio_hidden_states, torch.Tensor):
            return None
        a_sz = int(audio_hidden_states.shape[0])
        h_sz = int(hidden_states.shape[0])
        if a_sz != h_sz and queued_runtime_info:
            regathered = self._gather_last_hidden_per_request(audio_hidden_states, queued_runtime_info)
            if int(regathered.shape[0]) != a_sz:
                logger.warning(
                    "Re-gathered talker hidden for logits: before=%d after=%d expected=%d",
                    a_sz,
                    int(regathered.shape[0]),
                    h_sz,
                )
                audio_hidden_states = regathered
                a_sz = int(audio_hidden_states.shape[0])
        if a_sz > h_sz:
            # Last-resort single-row trim. If the queue is still misaligned here,
            # fail closed instead of feeding cross-request hidden states to TTS.
            audio_hidden_states = audio_hidden_states[-h_sz:]
            a_sz = int(audio_hidden_states.shape[0])
        if a_sz != h_sz:
            logger.warning(
                "Dropping mismatched talker hidden for logits: hidden_rows=%d logits_rows=%d",
                a_sz,
                h_sz,
            )
            return None
        return audio_hidden_states

    def _is_all_audio_only(self, queued_runtime_info: list[Any]) -> bool:
        if not queued_runtime_info:
            return False
        return all(
            self._normalize_output_mode(info if isinstance(info, dict) else None) == "audio_only"
            for info in queued_runtime_info
        )

    def _build_audio_only_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_thinker = self._get_text_logits_hidden(hidden_states, hidden_states)
        logits = torch.full(
            (hidden_states.shape[0], self.vocab_size),
            float("-inf"),
            dtype=normed_thinker.dtype,
            device=normed_thinker.device,
        )
        special_ids = self._audio_only_allowed_text_token_ids
        if not special_ids:
            return logits

        lm_weight = self.lm_head.weight
        normed = normed_thinker.squeeze(1) if normed_thinker.dim() == 3 else normed_thinker
        for token_id in special_ids:
            if token_id == self.audio_end_token_id:
                continue
            if 0 <= token_id < lm_weight.shape[0]:
                logits[:, token_id] = torch.mv(normed, lm_weight[token_id])
        return logits

    def _compute_base_logits(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor | None,
        queued_runtime_info: list[Any],
    ) -> torch.Tensor | None:
        use_split = (
            isinstance(audio_hidden_states, torch.Tensor) and audio_hidden_states.shape[0] == hidden_states.shape[0]
        )
        if use_split and self._is_all_audio_only(queued_runtime_info):
            return self._build_audio_only_logits(hidden_states)
        if use_split:
            normed_thinker = self._get_text_logits_hidden(hidden_states, hidden_states)
            return self.logits_processor(self.lm_head, normed_thinker)
        return self.logits_processor(self.lm_head, hidden_states)

    def _resolve_row_runtime_info(
        self,
        runtime_info: list[Any],
        row_count: int,
    ) -> list[Any] | None:
        if not runtime_info:
            return None
        if len(runtime_info) == row_count:
            return runtime_info
        if len(runtime_info) == 1:
            return [runtime_info[0]] * row_count

        modes = [self._normalize_output_mode(info if isinstance(info, dict) else None) for info in runtime_info]
        if modes and all(mode == modes[0] for mode in modes):
            logger.warning(
                "Runtime_info/logits row mismatch: runtime=%d logits=%d; broadcasting mode=%s",
                len(runtime_info),
                row_count,
                modes[0],
            )
            return [runtime_info[-1]] * row_count
        return None

    def _resolve_row_sampling_state(
        self,
        output_token_ids: Any,
        row_idx: int,
        row_count: int,
    ) -> tuple[list[int], bool]:
        is_sampled_row = True
        if isinstance(output_token_ids, list) and output_token_ids and len(output_token_ids) == 1 and row_count > 1:
            is_sampled_row = row_idx == row_count - 1

        row_output_ids: list[int] = []
        if isinstance(output_token_ids, list) and output_token_ids:
            if len(output_token_ids) == row_count and row_idx < len(output_token_ids):
                row_output_ids = output_token_ids[row_idx]
            elif len(output_token_ids) == 1:
                row_output_ids = output_token_ids[0]
            elif row_idx < len(output_token_ids):
                row_output_ids = output_token_ids[row_idx]
        return row_output_ids, is_sampled_row

    def _apply_row_mode_adjustments(
        self,
        *,
        logits: torch.Tensor,
        row_runtime_info: list[Any],
        output_token_ids: Any,
        audio_hidden_states: torch.Tensor | None,
    ) -> None:
        row_count = len(row_runtime_info)
        for row_idx, req_info_raw in enumerate(row_runtime_info):
            req_info = req_info_raw if isinstance(req_info_raw, dict) else None
            mode = self._normalize_output_mode(req_info)

            force_audio_first_token = False
            req_runtime_id: str | None = None
            req_state: AudioDecodeState | None = None
            if req_info is not None:
                raw_flag = unwrap_singleton_list(req_info.get("force_audio_first_token", False))
                force_audio_first_token = bool(raw_flag)
                # Read continuation_silence_frames from request info
                csf_raw = unwrap_singleton_list(req_info.get("continuation_silence_frames", 0))
                csf = int(csf_raw) if csf_raw else 0
                req_runtime_id = normalize_runtime_request_id(
                    req_info.get("global_request_id", req_info.get("_omni_req_id"))
                )
                req_state = self._get_audio_decode_state(req_runtime_id)
                if req_state is not None and req_state.continuation_silence_frames == 0 and csf > 0:
                    req_state.continuation_silence_frames = csf

            req_debug_id = (
                req_info.get("global_request_id", req_info.get("_omni_req_id")) if req_info is not None else None
            )
            logger.debug(
                "Row=%d req_id=%s mode=%s",
                row_idx,
                req_debug_id,
                mode,
            )
            if isinstance(output_token_ids, list) and row_idx < len(output_token_ids) and not output_token_ids[row_idx]:
                top_vals, top_ids = torch.topk(logits[row_idx].float(), k=min(20, int(logits.shape[-1])))
                logger.debug(
                    "First-step row=%d req_id=%s top_ids=%s top_vals=%s",
                    row_idx,
                    req_debug_id,
                    top_ids.tolist(),
                    [round(float(v), 4) for v in top_vals.tolist()],
                )

            row_output_ids, is_sampled_row = self._resolve_row_sampling_state(
                output_token_ids=output_token_ids,
                row_idx=row_idx,
                row_count=row_count,
            )
            if mode == "audio_only":
                self._generate_audio_for_row(
                    logits=logits,
                    row_idx=row_idx,
                    req_info=req_info,
                    req_runtime_id=req_runtime_id,
                    req_state=req_state,
                    force_audio_first_token=force_audio_first_token,
                    is_sampled_row=is_sampled_row,
                    row_output_ids=row_output_ids,
                    audio_hidden_states=audio_hidden_states,
                )
            else:
                self._mask_audio_logits_for_text_mode(logits, row_idx)

    def _suppress_first_step_eos(self, logits: torch.Tensor, output_token_ids: Any) -> None:
        eos_token_id = self.eos_token_id
        if not isinstance(eos_token_id, int):
            return
        if not (0 <= eos_token_id < logits.shape[-1]):
            return
        if not isinstance(output_token_ids, list):
            return
        if len(output_token_ids) != logits.shape[0]:
            return
        for row_idx, out_ids in enumerate(output_token_ids):
            if not out_ids:
                logits[row_idx, eos_token_id] = torch.finfo(logits.dtype).tiny

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: SamplingMetadata = None):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if not isinstance(hidden_states, torch.Tensor):
            return None

        queued_runtime_info = self._peek_runtime_info()
        audio_hidden_states = self._pop_audio_hidden_for_logits(hidden_states, queued_runtime_info)
        logits = self._compute_base_logits(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            queued_runtime_info=queued_runtime_info,
        )

        if logits is None:
            return None

        if int(logits.shape[-1]) > int(self.vocab_size):
            logits[:, int(self.vocab_size) :] = float("-inf")

        runtime_info = self._runtime_info_queue.popleft() if len(self._runtime_info_queue) > 0 else []
        if not isinstance(runtime_info, list):
            runtime_info = []
        output_token_ids = (
            getattr(sampling_metadata, "output_token_ids", None) if sampling_metadata is not None else None
        )
        row_runtime_info = self._resolve_row_runtime_info(runtime_info, int(logits.shape[0]))
        if row_runtime_info is not None:
            self._apply_row_mode_adjustments(
                logits=logits,
                row_runtime_info=row_runtime_info,
                output_token_ids=output_token_ids,
                audio_hidden_states=audio_hidden_states,
            )
        self._suppress_first_step_eos(logits, output_token_ids)

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: object | None = None,
        **info_dict: Any,
    ) -> dict[str, Any]:
        del multimodal_outputs
        req_id = (
            normalize_runtime_request_id(info_dict.get("global_request_id", info_dict.get("_omni_req_id")))
            or "__unknown__"
        )
        self._get_audio_decode_state(None if req_id == "__unknown__" else req_id)
        if len(self._talker_hidden_for_postprocess_queue) > 0:
            hidden_states = self._talker_hidden_for_postprocess_queue.popleft()
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.numel() == 0:
            return {}
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)
        latest_hidden = hidden_states[-1:].detach().contiguous().clone()
        return {"prev_hidden": latest_hidden}

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return strip_raon_audio_markers(text)

    @torch.inference_mode()
    def get_audio_output_embeds(
        self,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if audio_codes.ndim != 3 or audio_codes_mask.ndim != 2:
            raise ValueError(
                f"Expected audio_codes [B,T,G] and mask [B,T], "
                f"got {tuple(audio_codes.shape)} and "
                f"{tuple(audio_codes_mask.shape)}"
            )
        latent = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        adaptor_out = self.output_adaptor(
            latent,
            mask=audio_codes_mask,
        )
        return adaptor_out.outputs_embeds, adaptor_out.mask

    @torch.inference_mode()
    def generate_audio_codes(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        num_code_groups: int,
        speaker_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """All RVQ groups from layer-0 token ids + ``proj_code`` hidden state → ``[B, num_code_groups]``."""
        if num_code_groups == 1:
            return input_ids

        input_ids = input_ids.to(torch.long)
        last_hidden = inputs_embeds
        if speaker_embeds is not None and self.proj_speaker_code is not None:
            if speaker_embeds.ndim == 2:
                speaker_embeds = speaker_embeds.unsqueeze(1)
            if speaker_embeds.ndim != 3:
                raise AssertionError(
                    "speaker_embeds must be 2D/3D tensor for code predictor conditioning, "
                    f"got shape={tuple(speaker_embeds.shape)}."
                )
            if int(speaker_embeds.shape[0]) != int(inputs_embeds.shape[0]):
                raise AssertionError(
                    "speaker_embeds batch size must match audio-code rows: "
                    f"speaker_batch={int(speaker_embeds.shape[0])}, code_rows={int(inputs_embeds.shape[0])}."
                )
            speaker_embeds = speaker_embeds.to(
                device=inputs_embeds.device,
                dtype=self.proj_speaker_code.weight.dtype,
            )
            speaker_cond = self.proj_speaker_code(speaker_embeds.squeeze(1)).to(dtype=inputs_embeds.dtype)
            last_hidden = inputs_embeds + speaker_cond.unsqueeze(1)

        layer0_code = input_ids.reshape(-1)
        bsz = layer0_code.shape[0]

        # Layer-0 RVQ embeddings from the code predictor's first embedding table.
        codec_embeds = self.code_predictor.get_input_embeddings()
        layer0_embed = codec_embeds[0](layer0_code.unsqueeze(-1)).squeeze(1)

        return self.code_predictor.predict_codes(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_hidden=last_hidden.reshape(bsz, -1),
        )

    def on_requests_finished(self, req_ids: list[str]) -> None:
        for req_id in req_ids:
            self._ras.cleanup(req_id)
            self._audio_decode_state.pop(req_id, None)
            # Clean up stage input processor async-chunk accumulator state.
            try:
                from vllm_omni.model_executor.stage_input_processors.raon import async_chunk_cleanup_request

                async_chunk_cleanup_request(req_id)
            except ImportError:
                pass

    def cleanup_request_state(self, req_id: str) -> None:
        self.on_requests_finished([req_id])

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Explicit skip_prefixes avoids silently loading weights for absent optional modules.
        skip_prefixes: list[str] = []

        # Skip speaker weights when module is not built (checkpoint may omit them).
        if getattr(self, "speaker_encoder", None) is None:
            skip_prefixes.append("speaker_encoder.")
        if getattr(self, "proj_speaker_code", None) is None:
            if bool(self.config.speaker_embedding_to_code_predictor):
                logger.warning(
                    "Config expects speaker_embedding_to_code_predictor, "
                    "but proj_speaker_code is not instantiated; skipping proj_speaker_code.* weights."
                )
            skip_prefixes.append("proj_speaker_code.")
        skip_prefixes.append("text_output_norm.")

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
            ignore_unexpected_suffixes=["cluster_usage", "embed_sum", "initialized"],
        )
        prefix_names = (
            "text_model.",
            "talker.",
            "lm_head.",
            "audio_lm_head.",
            "text_output_norm.",
            "thinker_to_talker_proj.",
            "audio_encoder.",
            "input_adaptor.",
            "output_adaptor.",
            "proj_code.",
            "proj_speaker_code.",
            "code_predictor.",
            "speaker_encoder.",
            "audio_tokenizer.",
        )
        seen_counts: dict[str, int] = defaultdict(int)

        def tracked_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                seen_counts["__total__"] += 1
                for prefix in prefix_names:
                    if name.startswith(prefix):
                        seen_counts[prefix] += 1
                        break

                # audio_lm_head.weight loads directly into self.audio_lm_head via AutoWeightsLoader
                yield name, tensor

        loaded = loader.load_weights(tracked_weights())

        # Talker uses projected embeddings, not token ids; mark embed_tokens as loaded to satisfy vLLM init check.
        loaded.discard("talker.embed_tokens.weight")
        loaded.add("talker.embed_tokens.weight")

        loaded_counts: dict[str, int] = defaultdict(int)
        for name in loaded:
            for prefix in prefix_names:
                if name.startswith(prefix):
                    loaded_counts[prefix] += 1
                    break

        logger.info(
            "Loaded weights: stage=%s ckpt_total=%d loaded=%d "
            "text_model=%d/%d talker=%d/%d lm_head=%d/%d "
            "audio_encoder=%d/%d audio_tokenizer=%d/%d "
            "code_predictor=%d/%d output_adaptor=%d/%d input_adaptor=%d/%d "
            "proj_code=%d/%d thinker_to_talker_proj=%d/%d",
            self.model_stage,
            int(seen_counts.get("__total__", 0)),
            int(len(loaded)),
            int(loaded_counts.get("text_model.", 0)),
            int(seen_counts.get("text_model.", 0)),
            int(loaded_counts.get("talker.", 0)),
            int(seen_counts.get("talker.", 0)),
            int(loaded_counts.get("lm_head.", 0)),
            int(seen_counts.get("lm_head.", 0)),
            int(loaded_counts.get("audio_encoder.", 0)),
            int(seen_counts.get("audio_encoder.", 0)),
            int(loaded_counts.get("audio_tokenizer.", 0)),
            int(seen_counts.get("audio_tokenizer.", 0)),
            int(loaded_counts.get("code_predictor.", 0)),
            int(seen_counts.get("code_predictor.", 0)),
            int(loaded_counts.get("output_adaptor.", 0)),
            int(seen_counts.get("output_adaptor.", 0)),
            int(loaded_counts.get("input_adaptor.", 0)),
            int(seen_counts.get("input_adaptor.", 0)),
            int(loaded_counts.get("proj_code.", 0)),
            int(seen_counts.get("proj_code.", 0)),
            int(loaded_counts.get("thinker_to_talker_proj.", 0)),
            int(seen_counts.get("thinker_to_talker_proj.", 0)),
        )

        # Truncate RoPE cos/sin caches to bf16 precision to match training. default: f32
        truncated = 0
        for module in self.modules():
            if hasattr(module, "cos_sin_cache") and isinstance(module.cos_sin_cache, torch.Tensor):
                cache = module.cos_sin_cache
                module.cos_sin_cache = cache.to(torch.bfloat16).to(cache.dtype)
                truncated += 1
        if truncated:
            logger.info("Truncated %d RoPE cos_sin_cache buffers to bf16 precision", truncated)

        return loaded


# Backward compatibility for earlier references in this branch.
RaonForConditionalGeneration = RaonModel
