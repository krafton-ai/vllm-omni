# SPDX-License-Identifier: Apache-2.0
"""Consolidated tests for Raon chat template builder and serving hooks.

Merges: test_chat_template_builder, test_serving_hooks.

Cuts applied:
- TestAudioLossRegression deleted (duplicates TestMultiTurnTextThenAudio)
- TestE2ENonICL / TestE2ENonICLvsICLContrast deleted (redundant with unit tests)
- Single-attribute task tests merged into parametrized
- TestApplyTaskDefaults 15 methods → 4 parametrized (one per task type)
- TestEstimateTts{Min,Max}Tokens → parametrized input/expected pairs
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.raon.serving_utils import (
    _GLOBAL_TOP_K,
    _TASK_DEFAULTS,
    RaonServingHooks,
)
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_PLACEHOLDER_SEQ,
    AUDIO_START,
    AUDIO_OUTPUT_OPEN_SEQ,
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_START_TOKEN,
    IM_END,
    IM_START,
    LEGACY_AUDIO_PAD_TOKEN,
    LEGACY_AUDIO_PLACEHOLDER_SEQ,
    OutputMode,
    RaonChatTemplateBuilder,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    TaskType,
    USER_PROMPT_MARKER,
    count_audio_placeholders_str,
    inject_placeholders_into_str,
    inject_placeholders_into_token_ids,
    normalize_placeholders_str,
    normalize_token_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ===================================================================
# Chat Template Builder: stubs & helpers
# ===================================================================


class _StubChatTokenizer:
    """Minimal tokenizer for chat template builder tests."""

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{IM_START.text}{role}\n{content}{IM_END.text}\n")
        if add_generation_prompt:
            parts.append(f"{IM_START.text}assistant\n")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_map = {
            IM_START.text: IM_START.id,
            IM_END.text: IM_END.id,
            AUDIO_START.text: AUDIO_START.id,
            AUDIO_END.text: AUDIO_END.id,
            AUDIO_INPUT_PLACEHOLDER.text: AUDIO_INPUT_PLACEHOLDER.id,
            AUDIO_OUTPUT_PLACEHOLDER.text: AUDIO_OUTPUT_PLACEHOLDER.id,
            SPEAKER_EMBEDDING_PLACEHOLDER.text: SPEAKER_EMBEDDING_PLACEHOLDER.id,
            LEGACY_AUDIO_PAD_TOKEN: 151673,
        }
        ids: list[int] = []
        remaining = text
        while remaining:
            matched = False
            for token_text, token_id in sorted(
                token_map.items(), key=lambda x: -len(x[0])
            ):
                if remaining.startswith(token_text):
                    ids.append(token_id)
                    remaining = remaining[len(token_text):]
                    matched = True
                    break
            if not matched:
                ids.append(ord(remaining[0]))
                remaining = remaining[1:]
        return ids


def _builder(tokenizer=None) -> RaonChatTemplateBuilder:
    return RaonChatTemplateBuilder(tokenizer=tokenizer or _StubChatTokenizer())


def _assert_chat_frame(prompt: str) -> None:
    assert f"{IM_START.text}user" in prompt
    assert IM_END.text in prompt
    assert f"{IM_START.text}assistant\n" in prompt


PH = AUDIO_PLACEHOLDER_SEQ

_AUDIO_INPUT_TASKS = {TaskType.STT, TaskType.SPOKEN_QA, TaskType.SPEECH_QA}
_TEXT_CONTENT_TASKS = {TaskType.TEXT_QA, TaskType.TTS, TaskType.SPEECH_QA}


def _kwargs_for_task(task: TaskType) -> dict:
    kwargs: dict = {"task": task}
    if task in _AUDIO_INPUT_TASKS:
        kwargs["audio_count"] = 1
    if task in _TEXT_CONTENT_TASKS or task.value == "tts_icl":
        kwargs["user_content"] = "test"
    return kwargs


# ===================================================================
# Single-turn tasks — parametrized
# ===================================================================


@pytest.mark.parametrize(
    "task,user_content,audio_count,expected_mode,expect_audio_input,expect_ph",
    [
        (TaskType.TEXT_QA, "Capital of France?", 0, OutputMode.TEXT_ONLY, False, False),
        (TaskType.STT, None, 1, OutputMode.TEXT_ONLY, True, True),
        (TaskType.TTS, "Hello world", 0, OutputMode.AUDIO_ONLY, False, False),
        (TaskType.SPOKEN_QA, None, 1, OutputMode.TEXT_ONLY, True, True),
        (TaskType.SPEECH_QA, "What is being said?", 1, OutputMode.TEXT_ONLY, True, True),
    ],
    ids=["text_qa", "stt", "tts", "spoken_qa", "speech_qa"],
)
def test_single_turn_task_basics(task, user_content, audio_count, expected_mode, expect_audio_input, expect_ph):
    kwargs: dict = {"task": task}
    if user_content:
        kwargs["user_content"] = user_content
    if audio_count:
        kwargs["audio_count"] = audio_count
    result = _builder().build_prompt(**kwargs)

    _assert_chat_frame(result.prompt_text)
    assert result.output_mode == expected_mode
    assert result.has_audio_input is expect_audio_input
    if expect_ph:
        assert PH in result.prompt_text
    else:
        assert PH not in result.prompt_text


# ===================================================================
# TextQA specifics
# ===================================================================


class TestTextQA:
    def test_multi_turn(self):
        result = _builder().build_prompt(
            task=TaskType.TEXT_QA,
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        )
        _assert_chat_frame(result.prompt_text)
        assert "Hi" in result.prompt_text
        assert "How are you?" in result.prompt_text


# ===================================================================
# STT specifics
# ===================================================================


class TestSTT:
    def test_instruction(self):
        result = _builder().build_prompt(task=TaskType.STT, audio_count=1, instruction="Transcribe this audio.")
        assert "Transcribe this audio." in result.prompt_text

    def test_placeholder_before_instruction(self):
        result = _builder().build_prompt(task=TaskType.STT, audio_count=1, instruction="Transcribe this audio.")
        assert result.prompt_text.index(PH) < result.prompt_text.index("Transcribe this audio.")

    def test_exactly_one_placeholder(self):
        result = _builder().build_prompt(task=TaskType.STT, audio_count=1)
        assert result.prompt_text.count(PH) == 1


# ===================================================================
# TTS specifics
# ===================================================================


class TestTTS:
    def test_default_is_instruction_style(self):
        result = _builder().build_prompt(task=TaskType.TTS, user_content="Hello")
        assert "Speak the following text:" in result.prompt_text

    @pytest.mark.parametrize(
        "style,expect_instruction",
        [("instruction", True), ("raw", False)],
    )
    def test_prompt_style(self, style, expect_instruction):
        result = _builder().build_prompt(
            task=TaskType.TTS, user_content="Hello", tts_prompt_style=style,
        )
        assert ("Speak the following text:" in result.prompt_text) is expect_instruction

    def test_audio_start_suffix(self):
        result = _builder().build_prompt(
            task=TaskType.TTS, user_content="Test", append_audio_start=True,
        )
        assert result.prompt_text.endswith(AUDIO_START.text)

    def test_no_audio_start_when_disabled(self):
        result = _builder().build_prompt(
            task=TaskType.TTS, user_content="Test", append_audio_start=False,
        )
        assert not result.prompt_text.endswith(AUDIO_START.text)

    def test_speaker_token_before_instruction(self):
        result = _builder().build_prompt(
            task=TaskType.TTS, user_content="Hello", prepend_speaker_token=True,
        )
        spk = SPEAKER_EMBEDDING_PLACEHOLDER.text
        assert spk in result.prompt_text
        assert result.prompt_text.index(spk) < result.prompt_text.index("Speak the following text:")

    def test_exact_layout_matches_original(self):
        spk = SPEAKER_EMBEDDING_PLACEHOLDER.text
        result = _builder().build_prompt(
            task=TaskType.TTS,
            user_content="Hello world",
            prepend_speaker_token=True,
            append_audio_start=True,
        )
        expected_user_content = f"{spk}Speak the following text:\nHello world"
        expected = (
            f"{IM_START.text}user\n{expected_user_content}{IM_END.text}\n"
            f"{IM_START.text}assistant\n"
            f"{AUDIO_START.text}"
        )
        assert result.prompt_text == expected

    def test_no_audio_input_placeholder(self):
        result = _builder().build_prompt(task=TaskType.TTS, user_content="Hello")
        assert PH not in result.prompt_text


# ===================================================================
# SpeechQA specifics
# ===================================================================


class TestSpeechQA:
    def test_placeholder_before_text(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA, user_content="Describe this.", audio_count=1,
        )
        assert result.prompt_text.index(PH) < result.prompt_text.index("Describe this.")

    def test_multiple_audios(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA, user_content="Compare.", audio_count=2,
        )
        assert result.prompt_text.count(PH) == 2

    def test_no_duplicate_placeholder(self):
        content = f"{PH}What is this?"
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA, user_content=content, audio_count=1,
        )
        assert result.prompt_text.count(PH) == 1


# ===================================================================
# Multi-turn interleaving
# ===================================================================


class TestMultiTurnTextThenAudio:
    def test_text_then_audio_with_question(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What did I say?", "audio_count": 1},
            ],
        )
        _assert_chat_frame(result.prompt_text)
        assert result.prompt_text.count(PH) == 1
        assert "What did I say?" in result.prompt_text
        assert result.has_audio_input is True

    def test_text_then_audio_placeholder_in_correct_turn(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "My color is blue. Remember."},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": "What is my favorite color?", "audio_count": 1},
            ],
        )
        text = result.prompt_text
        ph_pos = text.index(PH)
        t1_pos = text.index("My color is blue.")
        t2_pos = text.index("What is my favorite color?")
        assert ph_pos > t1_pos
        assert ph_pos < t2_pos + len("What is my favorite color?")

    def test_text_then_audio_three_turns(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "My color is blue. Remember."},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4."},
                {"role": "user", "content": "What is my favorite color from T1?", "audio_count": 1},
            ],
        )
        assert result.prompt_text.count(PH) == 1
        assert "What is my favorite color from T1?" in result.prompt_text
        assert result.has_audio_input is True


class TestMultiTurnAudioThenText:
    def test_audio_then_text(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "What does this say?", "audio_count": 1},
                {"role": "assistant", "content": "It says hello."},
                {"role": "user", "content": "Summarize it in one word."},
            ],
        )
        _assert_chat_frame(result.prompt_text)
        assert result.prompt_text.count(PH) == 1

    def test_audio_then_text_no_extra_placeholder(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "", "audio_count": 1},
                {"role": "assistant", "content": "I heard something."},
                {"role": "user", "content": "What was it?"},
            ],
        )
        assert result.prompt_text.count(PH) == 1


class TestMultiTurnAudioThenAudio:
    def test_two_audio_turns(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "Listen.", "audio_count": 1},
                {"role": "assistant", "content": "I heard a bird."},
                {"role": "user", "content": "And this?", "audio_count": 1},
            ],
        )
        assert result.prompt_text.count(PH) == 2
        assert result.has_audio_input is True
        assert result.audio_count == 2


class TestMultiTurnComplexInterleaving:
    def test_five_turns_alternating(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "My color is blue."},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": "I live in Tokyo", "audio_count": 1},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": "Where do I live? What was my color?"},
            ],
        )
        _assert_chat_frame(result.prompt_text)
        assert result.prompt_text.count(PH) == 1

    def test_audio_text_audio_sequence(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "First clip", "audio_count": 1},
                {"role": "assistant", "content": "Heard it."},
                {"role": "user", "content": "Now explain what happened."},
                {"role": "assistant", "content": "A dog barked."},
                {"role": "user", "content": "Second clip", "audio_count": 1},
            ],
        )
        assert result.prompt_text.count(PH) == 2

    def test_multiple_audios_single_turn(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "Compare all three clips.", "audio_count": 3},
            ],
        )
        assert result.prompt_text.count(PH) == 3
        assert result.audio_count == 3

    def test_system_then_audio_turns(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "Translate", "audio_count": 1},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What about this?", "audio_count": 1},
            ],
            system_prompt="You are a translator.",
        )
        assert "You are a translator." in result.prompt_text
        assert result.prompt_text.count(PH) == 2

    def test_six_turn_real_world_scenario(self):
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "Here's a recording.", "audio_count": 1},
                {"role": "assistant", "content": "I heard you say 'I love pizza'."},
                {"role": "user", "content": "And this one too.", "audio_count": 1},
                {"role": "assistant", "content": "That was 'I live in Seoul'."},
                {"role": "user", "content": "What is my name and what did I say?"},
            ],
        )
        assert result.prompt_text.count(PH) == 2
        assert "My name is Alice." in result.prompt_text


# ===================================================================
# Legacy token normalisation
# ===================================================================


class TestLegacyTokenNormalisation:
    def test_normalize_legacy_placeholder_str(self):
        legacy = f"prefix{LEGACY_AUDIO_PLACEHOLDER_SEQ}suffix"
        normalized = normalize_placeholders_str(legacy)
        assert PH in normalized
        assert LEGACY_AUDIO_PAD_TOKEN not in normalized

    def test_count_includes_legacy(self):
        text = f"{PH}{LEGACY_AUDIO_PLACEHOLDER_SEQ}"
        assert count_audio_placeholders_str(text) == 2

    def test_count_no_double_counting(self):
        assert count_audio_placeholders_str(f"{PH}{PH}") == 2

    def test_legacy_in_multi_turn_normalised(self):
        legacy_content = f"{LEGACY_AUDIO_PLACEHOLDER_SEQ}Listen to this."
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[{"role": "user", "content": legacy_content, "audio_count": 1}],
        )
        assert LEGACY_AUDIO_PAD_TOKEN not in result.prompt_text
        assert result.prompt_text.count(PH) == 1


# ===================================================================
# Placeholder injection: str path
# ===================================================================


class TestInjectPlaceholdersStr:
    def test_inject_into_empty(self):
        result = inject_placeholders_into_str("hello", num_audios=1)
        assert result.count(PH) == 1

    def test_no_injection_when_enough(self):
        assert inject_placeholders_into_str(f"{PH}hello", num_audios=1).count(PH) == 1

    def test_inject_distributes_across_turns(self):
        prompt = (
            f"{IM_START.text}user\nTurn1{IM_END.text}\n"
            f"{IM_START.text}assistant\nOK{IM_END.text}\n"
            f"{IM_START.text}user\nTurn2{IM_END.text}\n"
        )
        assert inject_placeholders_into_str(prompt, num_audios=2).count(PH) == 2

    def test_inject_zero_is_noop(self):
        text = "hello"
        assert inject_placeholders_into_str(text, num_audios=0) == text

    def test_legacy_counted_and_normalised(self):
        text = f"{LEGACY_AUDIO_PLACEHOLDER_SEQ}hello"
        result = inject_placeholders_into_str(text, num_audios=1)
        assert result.count(PH) == 1
        assert LEGACY_AUDIO_PAD_TOKEN not in result


# ===================================================================
# Placeholder injection: token ID path
# ===================================================================


class TestInjectPlaceholdersTokenIds:
    PH_IDS = [AUDIO_START.id, AUDIO_INPUT_PLACEHOLDER.id, AUDIO_END.id]
    LEGACY_PH_IDS = [AUDIO_START.id, 151673, AUDIO_END.id]
    MARKER_IDS = [IM_START.id, ord("u"), ord("s"), ord("e"), ord("r"), ord("\n")]

    def test_inject_when_missing(self):
        result = inject_placeholders_into_token_ids(
            [1, 2, 3], num_audios=1, ph_ids=self.PH_IDS,
        )
        assert self.PH_IDS[0] in result

    def test_no_inject_when_present(self):
        ids = self.PH_IDS + [1, 2, 3]
        assert inject_placeholders_into_token_ids(ids, num_audios=1, ph_ids=self.PH_IDS) == ids

    def test_counts_legacy(self):
        ids = self.LEGACY_PH_IDS + [1, 2, 3]
        result = inject_placeholders_into_token_ids(
            ids, num_audios=1, ph_ids=self.PH_IDS, legacy_ph_ids=self.LEGACY_PH_IDS,
        )
        assert result == ids

    def test_inject_into_correct_turn(self):
        ids = (
            self.MARKER_IDS + [ord("H"), ord("i")]
            + [IM_END.id]
            + self.MARKER_IDS + [ord("Q")]
        )
        result = inject_placeholders_into_token_ids(
            ids, num_audios=1, ph_ids=self.PH_IDS, marker_ids=self.MARKER_IDS,
        )
        assert result.count(AUDIO_START.id) == 1

    def test_zero_is_noop(self):
        ids = [1, 2, 3]
        assert inject_placeholders_into_token_ids(ids, num_audios=0, ph_ids=self.PH_IDS) == ids


# ===================================================================
# normalize_token_ids
# ===================================================================


class TestNormalizeTokenIds:
    def test_replaces_output_with_input(self):
        ids = [1, AUDIO_OUTPUT_PLACEHOLDER.id, 3]
        result = normalize_token_ids(ids)
        assert AUDIO_OUTPUT_PLACEHOLDER.id not in result
        assert AUDIO_INPUT_PLACEHOLDER.id in result

    def test_ensures_placeholder_subsequence(self):
        ids = [1, AUDIO_INPUT_PLACEHOLDER.id, 3]
        result = normalize_token_ids(ids)
        assert AUDIO_START.id in result and AUDIO_END.id in result

    def test_no_change_when_already_correct(self):
        ids = [1, AUDIO_START.id, AUDIO_INPUT_PLACEHOLDER.id, AUDIO_END.id, 3]
        assert normalize_token_ids(ids) == ids

    def test_no_audio_tokens_passthrough(self):
        assert normalize_token_ids([1, 2, 3]) == [1, 2, 3]


# ===================================================================
# Tokenizer fallback
# ===================================================================


class TestTokenizerFallback:
    def test_fallback_without_apply_chat_template(self):
        class _NoTemplate:
            pass

        result = _builder(tokenizer=_NoTemplate()).build_prompt(
            task=TaskType.TEXT_QA, user_content="Hello",
        )
        _assert_chat_frame(result.prompt_text)

    def test_fallback_tts(self):
        class _NoTemplate:
            pass

        result = _builder(tokenizer=_NoTemplate()).build_prompt(
            task=TaskType.TTS, user_content="Speak", append_audio_start=True,
        )
        assert f"{IM_START.text}user\n" in result.prompt_text
        assert f"{IM_START.text}assistant\n" in result.prompt_text


# ===================================================================
# Token ID encoding
# ===================================================================


class TestBuildPromptTokenIds:
    def test_stt_contains_audio_ids(self):
        b = _builder()
        result = b.build_prompt(task=TaskType.STT, audio_count=1)
        ids = b.encode_prompt(result.prompt_text)
        assert AUDIO_START.id in ids
        assert AUDIO_INPUT_PLACEHOLDER.id in ids

    def test_text_qa_no_audio_ids(self):
        b = _builder()
        result = b.build_prompt(task=TaskType.TEXT_QA, user_content="Hello")
        ids = b.encode_prompt(result.prompt_text)
        assert AUDIO_INPUT_PLACEHOLDER.id not in ids

    def test_no_output_placeholder_leaked(self):
        b = _builder()
        for task in TaskType:
            result = b.build_prompt(**_kwargs_for_task(task))
            assert AUDIO_OUTPUT_PLACEHOLDER.text not in result.prompt_text


# ===================================================================
# PromptResult metadata — parametrized
# ===================================================================


class TestPromptResultMetadata:
    @pytest.mark.parametrize(
        "task,expected_mode",
        [
            (TaskType.TEXT_QA, OutputMode.TEXT_ONLY),
            (TaskType.STT, OutputMode.TEXT_ONLY),
            (TaskType.TTS, OutputMode.AUDIO_ONLY),
            (TaskType.SPOKEN_QA, OutputMode.TEXT_ONLY),
            (TaskType.SPEECH_QA, OutputMode.TEXT_ONLY),
        ],
    )
    def test_output_mode(self, task, expected_mode):
        assert _builder().build_prompt(**_kwargs_for_task(task)).output_mode == expected_mode

    def test_audio_input_flag(self):
        b = _builder()
        for task in TaskType:
            result = b.build_prompt(**_kwargs_for_task(task))
            assert result.has_audio_input is (task in _AUDIO_INPUT_TASKS)

    def test_task_preserved(self):
        b = _builder()
        for task in TaskType:
            assert b.build_prompt(**_kwargs_for_task(task)).task == task


# ===================================================================
# E2E: ICL TTS path (kept — unique ICL scenarios)
# ===================================================================


def _build_icl_prompt(
    target_text: str,
    ref_text: str,
    prepend_speaker_token: bool = True,
) -> str:
    spk = SPEAKER_EMBEDDING_PLACEHOLDER.text
    user_content = f"Speak the following text:\n{ref_text} {target_text}"
    if prepend_speaker_token and not user_content.startswith(spk):
        user_content = f"{spk}{user_content}"

    assistant_content = f"{AUDIO_START_TOKEN}{AUDIO_OUTPUT_PAD_TOKEN}"
    prompt = (
        f"{IM_START.text}user\n{user_content}{IM_END.text}\n"
        f"{IM_START.text}assistant\n{assistant_content}"
    )
    return prompt


class TestE2EICL:
    def test_icl_prompt_contains_output_open_seq(self):
        assert AUDIO_OUTPUT_OPEN_SEQ in _build_icl_prompt("Hello.", "Ref text.")

    def test_icl_prompt_no_input_placeholder(self):
        assert PH not in _build_icl_prompt("Hello.", "Ref text.")

    def test_icl_count_recognises_output_placeholder(self):
        assert count_audio_placeholders_str(_build_icl_prompt("Hello.", "Ref text.")) == 1

    def test_icl_inject_does_not_add_input_placeholder(self):
        prompt = _build_icl_prompt("Hello.", "Ref text.")
        after = inject_placeholders_into_str(prompt, num_audios=1)
        assert after == prompt

    def test_icl_prompt_exact_layout(self):
        spk = SPEAKER_EMBEDDING_PLACEHOLDER.text
        prompt = _build_icl_prompt("Target.", "Reference.")
        expected = (
            f"{IM_START.text}user\n"
            f"{spk}Speak the following text:\n"
            f"Reference. Target.{IM_END.text}\n"
            f"{IM_START.text}assistant\n"
            f"{AUDIO_START.text}{AUDIO_OUTPUT_PLACEHOLDER.text}"
        )
        assert prompt == expected

    def test_icl_prompt_without_speaker(self):
        prompt = _build_icl_prompt("Hello.", "Ref.", prepend_speaker_token=False)
        assert SPEAKER_EMBEDDING_PLACEHOLDER.text not in prompt
        assert AUDIO_OUTPUT_OPEN_SEQ in prompt

    def test_icl_inject_with_legacy_and_output(self):
        prompt = (
            f"{IM_START.text}user\n{LEGACY_AUDIO_PLACEHOLDER_SEQ}Hello{IM_END.text}\n"
            f"{IM_START.text}assistant\n{AUDIO_OUTPUT_OPEN_SEQ}"
        )
        after = inject_placeholders_into_str(prompt, num_audios=2)
        assert count_audio_placeholders_str(after) >= 2


# ===================================================================
# Serving Hooks: stubs
# ===================================================================


class _StubHfConfig:
    def __init__(
        self,
        *,
        im_end_token_id: int | None = 151645,
        audio_end_token_id: int | None = 151670,
        im_start_token_id: int | None = None,
        audio_start_token_id: int | None = None,
    ) -> None:
        self.im_end_token_id = im_end_token_id
        self.audio_end_token_id = audio_end_token_id
        self.im_start_token_id = im_start_token_id
        self.audio_start_token_id = audio_start_token_id


class _StubModelConfig:
    def __init__(self, hf_config: _StubHfConfig | None = None) -> None:
        self.hf_config = hf_config
        self.model = None
        self.trust_remote_code = False


class _StubServingTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def _make_hooks(
    im_end_token_id: int | None = 151645,
    audio_end_token_id: int | None = 151670,
) -> RaonServingHooks:
    hf = _StubHfConfig(
        im_end_token_id=im_end_token_id,
        audio_end_token_id=audio_end_token_id,
    )
    return RaonServingHooks(_StubModelConfig(hf))


# ===================================================================
# Serving Hooks: build_tts_prompt
# ===================================================================


class TestBuildTtsPrompt:
    @pytest.mark.asyncio
    async def test_raw_style_no_speaker_token(self):
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hello world")
        assert "Hello world" in result
        assert "<|im_start|>user" in result

    @pytest.mark.asyncio
    async def test_instruction_style_via_env(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_PROMPT_STYLE", "instruction")
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hello world")
        assert "Speak the following text:" in result

    @pytest.mark.asyncio
    async def test_raw_style_is_default(self, monkeypatch):
        monkeypatch.delenv("RAON_TTS_PROMPT_STYLE", raising=False)
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hello world")
        assert "Speak the following text:" not in result

    @pytest.mark.asyncio
    async def test_prepend_speaker_token_adds_prefix(self):
        from vllm_omni.tokenizers.raon_tokenizer import SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN

        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hi", prepend_speaker_token=True)
        assert SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN in result

    @pytest.mark.asyncio
    async def test_prepend_speaker_token_not_duplicated(self):
        from vllm_omni.tokenizers.raon_tokenizer import SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN

        hooks = _make_hooks()
        text_with_token = f"{SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN}Hi"
        result = await hooks.build_tts_prompt(text_with_token, prepend_speaker_token=True)
        assert result.count(SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN) == 1

    @pytest.mark.asyncio
    async def test_tokenizer_apply_chat_template_used_when_available(self):
        class _FakeEngineClient:
            async def get_tokenizer(self):
                return _StubServingTokenizer()

        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Test text", engine_client=_FakeEngineClient())
        assert "Test text" in result

    @pytest.mark.asyncio
    async def test_fallback_template_when_no_engine_client(self):
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Fallback text")
        assert result == "<|im_start|>user\nFallback text<|im_end|>\n<|im_start|>assistant\n"

    @pytest.mark.asyncio
    async def test_engine_client_exception_falls_back_gracefully(self):
        class _BrokenEngineClient:
            async def get_tokenizer(self):
                raise RuntimeError("tokenizer unavailable")

        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Graceful", engine_client=_BrokenEngineClient())
        assert "Graceful" in result


# ===================================================================
# Serving Hooks: estimate_tts_min_tokens — parametrized
# ===================================================================


class TestEstimateTtsMinTokens:
    @pytest.mark.parametrize(
        "text,max_tokens,expected",
        [
            ("", None, 6),
            ("hi", None, 6),
            ("one two three four five six seven eight nine ten", None, 20),
            ("one two three four five six seven eight nine ten", 10, 9),
            ("hello world", 1, 0),
            ("", 0, 0),
            ("  hello   world  ", None, 4),
        ],
        ids=["empty", "short", "longer-text", "max-tokens-cap", "very-small-max", "zero-max", "extra-whitespace"],
    )
    def test_estimation(self, text, max_tokens, expected):
        hooks = _make_hooks()
        result = hooks.estimate_tts_min_tokens(text, max_tokens=max_tokens)
        if text == "  hello   world  ":
            clean = hooks.estimate_tts_min_tokens("hello world", max_tokens=None)
            assert result == clean
        else:
            assert result == expected

    def test_result_never_negative(self):
        assert _make_hooks().estimate_tts_min_tokens("", max_tokens=0) >= 0


# ===================================================================
# Serving Hooks: estimate_tts_max_tokens — parametrized
# ===================================================================


class TestEstimateTtsMaxTokens:
    @pytest.mark.parametrize(
        "text,hard_cap,expected",
        [
            ("", None, 48),
            ("hi", None, 48),
            ("one two three four", None, 64),
            (" ".join(["word"] * 100), None, 832),
            ("one two three four", 50, 50),
        ],
        ids=["empty", "short", "medium", "100-words", "custom-hard-cap"],
    )
    def test_estimation(self, text, hard_cap, expected):
        hooks = _make_hooks()
        kwargs = {"hard_cap": hard_cap} if hard_cap else {}
        assert hooks.estimate_tts_max_tokens(text, **kwargs) == expected

    def test_env_override_respected(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", "200")
        assert _make_hooks().estimate_tts_max_tokens("some text") == 200

    def test_env_override_clamped_to_hard_cap(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", "9999")
        assert _make_hooks().estimate_tts_max_tokens("some text", hard_cap=300) == 300

    @pytest.mark.parametrize("env_val", ["not_a_number", ""])
    def test_env_override_invalid_falls_back(self, monkeypatch, env_val):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", env_val)
        assert _make_hooks().estimate_tts_max_tokens("one two three four") == 64

    def test_result_at_least_one(self):
        assert _make_hooks().estimate_tts_max_tokens("", hard_cap=1) >= 1

    def test_env_override_at_least_one(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", "0")
        assert _make_hooks().estimate_tts_max_tokens("text") >= 1


# ===================================================================
# Serving Hooks: apply_sampling_parity
# ===================================================================


class TestApplySamplingParity:
    def _make_params(self, **kwargs) -> dict:
        base = {
            "stop": ["<extra_stop>"],
            "stop_token_ids": [9999],
            "ignore_eos": True,
            "min_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        }
        base.update(kwargs)
        return base

    def test_empty_list_is_noop(self):
        _make_hooks().apply_sampling_parity([], comprehension_stage_index=0)

    def test_applies_to_correct_stage_index(self):
        params_0 = self._make_params()
        params_1 = self._make_params()
        _make_hooks().apply_sampling_parity([params_0, params_1], comprehension_stage_index=1)
        assert params_1["temperature"] == _TASK_DEFAULTS["tts"]["temperature"]
        assert params_0["temperature"] == 1.0

    def test_out_of_bounds_index_clamps_to_last(self):
        params_0 = self._make_params()
        params_1 = self._make_params()
        _make_hooks().apply_sampling_parity([params_0, params_1], comprehension_stage_index=99)
        assert params_1["temperature"] == _TASK_DEFAULTS["tts"]["temperature"]

    def test_negative_index_clamps_to_first(self):
        params_0 = self._make_params()
        _make_hooks().apply_sampling_parity([params_0], comprehension_stage_index=-5)
        assert params_0["temperature"] == _TASK_DEFAULTS["tts"]["temperature"]

    def test_audio_stop_and_eos_set(self):
        params = self._make_params()
        _make_hooks().apply_sampling_parity([params], comprehension_stage_index=0)
        assert params["stop"] == []
        assert 151645 in params["stop_token_ids"]
        assert 151670 in params["stop_token_ids"]
        assert params["ignore_eos"] is False
        assert params["min_tokens"] >= 1

    def test_tts_defaults_applied(self):
        params = self._make_params()
        _make_hooks().apply_sampling_parity([params], comprehension_stage_index=0)
        assert params["temperature"] == 1.2
        assert params["top_k"] == _GLOBAL_TOP_K

    def test_client_explicit_temperature_respected(self):
        params = self._make_params(temperature=0.9)
        _make_hooks().apply_sampling_parity([params], comprehension_stage_index=0)
        assert params["temperature"] == 0.9


# ===================================================================
# Serving Hooks: apply_task_defaults — parametrized per task type
# ===================================================================


class TestApplyTaskDefaults:
    def _make_params(self, **kwargs) -> dict:
        base = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        }
        base.update(kwargs)
        return base

    @pytest.mark.parametrize(
        "task,expected_temp,expected_top_k,expected_rep_pen,expected_max",
        [
            ("tts", 1.2, 50, 1.0, 256),
            ("stt", 0.2, 0, 1.0, 512),
            ("speechqa", 0.7, 0, 1.1, 2048),
            ("spokenqa", 0.7, 0, 1.1, None),
        ],
        ids=["tts", "stt", "speechqa", "spokenqa"],
    )
    def test_task_defaults(self, task, expected_temp, expected_top_k, expected_rep_pen, expected_max):
        params = self._make_params()
        _make_hooks().apply_task_defaults(params, task)
        assert params["temperature"] == expected_temp
        assert params["top_k"] == expected_top_k
        assert params["repetition_penalty"] == expected_rep_pen
        if expected_max is not None:
            assert params["max_tokens"] == expected_max

    @pytest.mark.parametrize(
        "override_key,override_val,task",
        [
            ("temperature", 0.5, "speechqa"),
            ("repetition_penalty", 1.5, "speechqa"),
            ("top_k", 100, "tts"),
        ],
    )
    def test_client_override_preserved(self, override_key, override_val, task):
        params = self._make_params(**{override_key: override_val})
        _make_hooks().apply_task_defaults(params, task)
        assert params[override_key] == override_val

    def test_unknown_task_type_is_noop(self):
        params = self._make_params()
        _make_hooks().apply_task_defaults(params, "unknown")
        assert params["temperature"] == 1.0


# ===================================================================
# Serving Hooks: stop token ID resolution
# ===================================================================


class TestStopTokenIdResolution:
    def test_explicit_im_end_and_audio_end_used(self):
        hooks = _make_hooks(im_end_token_id=100, audio_end_token_id=200)
        assert 100 in hooks._audio_stop_token_ids
        assert 200 in hooks._audio_stop_token_ids

    def test_im_end_derived_from_im_start_plus_one(self):
        hf = _StubHfConfig(
            im_end_token_id=None, audio_end_token_id=200, im_start_token_id=99,
        )
        hooks = RaonServingHooks(_StubModelConfig(hf))
        assert 100 in hooks._audio_stop_token_ids

    def test_audio_end_derived_from_audio_start_plus_one(self):
        hf = _StubHfConfig(
            im_end_token_id=100, audio_end_token_id=None, audio_start_token_id=169,
        )
        hooks = RaonServingHooks(_StubModelConfig(hf))
        assert 170 in hooks._audio_stop_token_ids

    def test_no_hf_config_results_in_empty_stop_ids(self):
        hooks = RaonServingHooks(_StubModelConfig(hf_config=None))
        assert hooks._audio_stop_token_ids == []
