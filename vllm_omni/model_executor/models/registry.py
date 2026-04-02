from vllm.model_executor.models.registry import (
    _VLLM_MODELS,
    _LazyRegisteredModel,
    _ModelRegistry,
)

_OMNI_MODELS = {
    "Qwen2_5OmniForConditionalGeneration": (
        "qwen2_5_omni",
        "qwen2_5_omni",
        "Qwen2_5OmniForConditionalGeneration",
    ),
    "Qwen2_5OmniThinkerModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),
    "Qwen2_5OmniTalkerModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_talker",
        "Qwen2_5OmniTalkerForConditionalGeneration",
    ),
    "Qwen2_5OmniToken2WavModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavForConditionalGenerationVLLM",
    ),
    "Qwen2_5OmniToken2WavDiTModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavModel",
    ),
    "Qwen2ForCausalLM_old": ("qwen2_5_omni", "qwen2_old", "Qwen2ForCausalLM"),  # need to discuss
    # Qwen3 Omni MoE models
    "Qwen3OmniMoeForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni",
        "Qwen3OmniMoeForConditionalGeneration",
    ),
    "Qwen3OmniMoeThinkerForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni_moe_thinker",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeTalkerForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni_moe_talker",
        "Qwen3OmniMoeTalkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeCode2Wav": (
        "qwen3_omni",
        "qwen3_omni_code2wav",
        "Qwen3OmniMoeCode2Wav",
    ),
    "CosyVoice3Model": (
        "cosyvoice3",
        "cosyvoice3",
        "CosyVoice3Model",
    ),
    "MammothModa2Qwen2ForCausalLM": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2Qwen2ForCausalLM",
    ),
    "MammothModa2ARForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ARForConditionalGeneration",
    ),
    "MammothModa2DiTPipeline": (
        "mammoth_moda2",
        "pipeline_mammothmoda2_dit",
        "MammothModa2DiTPipeline",
    ),
    "MammothModa2ForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    "Mammothmoda2Model": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    "Qwen3TTSForConditionalGeneration": (
        "qwen3_tts",
        "qwen3_tts_talker",
        "Qwen3TTSTalkerForConditionalGeneration",
    ),
    "Qwen3TTSTalkerForConditionalGeneration": (
        "qwen3_tts",
        "qwen3_tts_talker",
        "Qwen3TTSTalkerForConditionalGeneration",
    ),
    "Qwen3TTSCode2Wav": (
        "qwen3_tts",
        "qwen3_tts_code2wav",
        "Qwen3TTSCode2Wav",
    ),
    ## mimo_audio
    "MiMoAudioModel": (
        "mimo_audio",
        "mimo_audio",
        "MiMoAudioForConditionalGeneration",
    ),
    "MiMoAudioLLMModel": (
        "mimo_audio",
        "mimo_audio_llm",
        "MiMoAudioLLMForConditionalGeneration",
    ),
    "MiMoAudioToken2WavModel": (
        "mimo_audio",
        "mimo_audio_code2wav",
        "MiMoAudioToken2WavForConditionalGenerationVLLM",
    ),
    ## glm_image
    "GlmImageForConditionalGeneration": (
        "glm_image",
        "glm_image_ar",
        "GlmImageForConditionalGeneration",
    ),
    "OmniBagelForConditionalGeneration": (
        "bagel",
        "bagel",
        "OmniBagelForConditionalGeneration",
    ),
    "HunyuanImage3ForCausalMM": (
        "hunyuan_image3",
        "hunyuan_image3",
        "HunyuanImage3ForConditionalGeneration",
    ),
    ## fish_speech (Fish Speech S2 Pro)
    "FishSpeechSlowARForConditionalGeneration": (
        "fish_speech",
        "fish_speech_slow_ar",
        "FishSpeechSlowARForConditionalGeneration",
    ),
    "FishSpeechDACDecoder": (
        "fish_speech",
        "fish_speech_dac_decoder",
        "FishSpeechDACDecoder",
    ),
    ## Voxtral TTS
    "VoxtralTTSForConditionalGeneration": (
        "voxtral_tts",
        "voxtral_tts",
        "VoxtralTTSForConditionalGeneration",
    ),
    "VoxtralTTSAudioGeneration": (
        "voxtral_tts",
        "voxtral_tts_audio_generation",
        "VoxtralTTSAudioGenerationForConditionalGeneration",
    ),
    "VoxtralTTSAudioTokenizer": ("voxtral_tts", "voxtral_tts_audio_tokenizer", "VoxtralTTSAudioTokenizer"),
    "RaonModel": ("raon", "raon", "RaonModel"),
    "RaonCode2WavModel": ("raon", "raon_code2wav", "RaonCode2WavModel"),
}


_SERVING_HOOKS: dict[str, str] = {
    "raon": "vllm_omni.model_executor.models.raon.serving_utils:RaonServingHooks",
}

_SERVING_CLASSES: dict[str, tuple[str, str]] = {
    "raon": (
        "vllm_omni.model_executor.models.raon.serving_utils:RaonOpenAIServingChat",
        "vllm_omni.model_executor.models.raon.serving_utils:RaonOpenAIServingSpeech",
    ),
}

_VLLM_OMNI_MODELS = {
    **_VLLM_MODELS,
    **_OMNI_MODELS,
}

OmniModelRegistry = _ModelRegistry(
    {
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm.model_executor.models.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
        },
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items()
        },
    }
)


def resolve_model_type(vllm_config, stage_configs=None) -> str | None:
    """Resolve model type from vllm_config or stage_configs."""
    try:
        model_type = vllm_config.model_config.hf_config.model_type
        if model_type in _SERVING_HOOKS:
            return model_type
    except (AttributeError, TypeError):
        pass
    if stage_configs:
        for cfg in stage_configs:
            model_type = getattr(cfg, "model_type", None)
            if model_type and model_type in _SERVING_HOOKS:
                return model_type
    return None


def create_serving_hooks(model_type: str, model_config=None):
    """Create serving hooks instance for the given model type."""
    if model_type not in _SERVING_HOOKS:
        return None
    module_path, cls_name = _SERVING_HOOKS[model_type].rsplit(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(model_config)


def resolve_serving_classes(model_type: str | None) -> tuple[type | None, type | None]:
    """Return (ChatClass, SpeechClass) for *model_type*, or (None, None)."""
    if not model_type or model_type not in _SERVING_CLASSES:
        return None, None
    import importlib

    result = []
    for dotted in _SERVING_CLASSES[model_type]:
        mod_path, cls_name = dotted.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        result.append(getattr(mod, cls_name))
    return tuple(result)  # type: ignore[return-value]
