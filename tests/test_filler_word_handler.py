"""
Comprehensive tests for the Filler Word Interruption Handler.

Tests all scenarios from the challenge:
1. User filler while agent speaks → ignored
2. User real interruption → allowed
3. User filler while agent quiet → registered
4. Mixed filler and command → allowed
5. Background murmur (low confidence) → ignored
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import RecognitionHooks
from livekit.agents.voice.filler_word_handler import (
    FillerWordHandler,
    FillerWordHandlerConfig,
)


class MockRecognitionHooks(RecognitionHooks):
    """Mock RecognitionHooks to track method calls."""

    def __init__(self) -> None:
        self.on_start_of_speech_calls: list = []
        self.on_vad_inference_done_calls: list = []
        self.on_end_of_speech_calls: list = []
        self.on_interim_transcript_calls: list = []
        self.on_final_transcript_calls: list = []
        self.on_end_of_turn_calls: list = []
        self.on_preemptive_generation_calls: list = []
        self.retrieve_chat_ctx_calls: int = 0

    def on_start_of_speech(self, ev) -> None:
        self.on_start_of_speech_calls.append(ev)

    def on_vad_inference_done(self, ev) -> None:
        self.on_vad_inference_done_calls.append(ev)

    def on_end_of_speech(self, ev) -> None:
        self.on_end_of_speech_calls.append(ev)

    def on_interim_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None) -> None:
        self.on_interim_transcript_calls.append((ev, speaking))

    def on_final_transcript(self, ev: stt.SpeechEvent) -> None:
        self.on_final_transcript_calls.append(ev)

    def on_end_of_turn(self, info) -> bool:
        self.on_end_of_turn_calls.append(info)
        return True

    def on_preemptive_generation(self, info) -> None:
        self.on_preemptive_generation_calls.append(info)

    def retrieve_chat_ctx(self):
        self.retrieve_chat_ctx_calls += 1
        return MagicMock()


def create_speech_event(text: str, is_final: bool = True, confidence: float = 1.0) -> stt.SpeechEvent:
    """Create a mock SpeechEvent for testing."""
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT if is_final else SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(text=text, language="en", confidence=confidence)],
    )


@pytest.fixture
def default_config() -> FillerWordHandlerConfig:
    """Default configuration for tests."""
    return FillerWordHandlerConfig(
        ignored_words=["uh", "umm", "hmm", "haan", "uhh", "um", "er", "ah"],
        interrupt_commands=["wait", "stop", "hold on", "cancel", "no", "wrong", "nevermind"],
        min_confidence_threshold=0.0,
        case_sensitive=False,
    )


@pytest.fixture
def mock_hooks() -> MockRecognitionHooks:
    """Create mock hooks for testing."""
    return MockRecognitionHooks()


@pytest.fixture
def handler(mock_hooks: MockRecognitionHooks, default_config: FillerWordHandlerConfig) -> FillerWordHandler:
    """Create a FillerWordHandler instance for testing."""
    return FillerWordHandler(base_hooks=mock_hooks, config=default_config)


class TestFillerWordDetection:
    """Test filler word detection logic."""

    def test_is_filler_only_single_word(self, handler: FillerWordHandler) -> None:
        """Test detection of single filler word."""
        assert handler._is_filler_only("uh") is True
        assert handler._is_filler_only("umm") is True
        assert handler._is_filler_only("hmm") is True

    def test_is_filler_only_multiple_words(self, handler: FillerWordHandler) -> None:
        """Test detection of multiple filler words."""
        assert handler._is_filler_only("uh umm hmm") is True
        assert handler._is_filler_only("umm uh") is True

    def test_is_filler_only_with_content(self, handler: FillerWordHandler) -> None:
        """Test that filler words with content are not filler-only."""
        assert handler._is_filler_only("uh hello") is False
        assert handler._is_filler_only("umm okay") is False
        assert handler._is_filler_only("hello uh") is False

    def test_is_filler_only_empty(self, handler: FillerWordHandler) -> None:
        """Test empty string handling."""
        assert handler._is_filler_only("") is False

    def test_contains_interrupt_command(self, handler: FillerWordHandler) -> None:
        """Test interrupt command detection."""
        assert handler._contains_interrupt_command("wait") is True
        assert handler._contains_interrupt_command("stop") is True
        assert handler._contains_interrupt_command("wait one second") is True
        assert handler._contains_interrupt_command("no not that one") is True
        assert handler._contains_interrupt_command("hello") is False

    def test_contains_interrupt_command_case_insensitive(
        self, handler: FillerWordHandler
    ) -> None:
        """Test case-insensitive interrupt command detection."""
        assert handler._contains_interrupt_command("WAIT") is True
        assert handler._contains_interrupt_command("Stop") is True
        assert handler._contains_interrupt_command("No") is True


class TestScenario1_FillerWhileAgentSpeaking:
    """Test Case 1: User filler while agent speaks → ignored."""

    def test_uh_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'uh' while agent is speaking is ignored."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("uh")

        handler.on_final_transcript(ev)

        # Should not forward to base hooks
        assert len(mock_hooks.on_final_transcript_calls) == 0

    def test_hmm_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'hmm' while agent is speaking is ignored."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("hmm")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 0

    def test_umm_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'umm' while agent is speaking is ignored."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("umm")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 0

    def test_multiple_fillers_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test multiple filler words while agent is speaking are ignored."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("uh umm hmm")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 0


class TestScenario2_RealInterruption:
    """Test Case 2: User real interruption → allowed immediately."""

    def test_wait_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'wait one second' while agent is speaking triggers interruption."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("wait one second")

        handler.on_final_transcript(ev)

        # Should forward to base hooks (allows interruption)
        assert len(mock_hooks.on_final_transcript_calls) == 1
        assert mock_hooks.on_final_transcript_calls[0].alternatives[0].text == "wait one second"

    def test_stop_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'stop' while agent is speaking triggers interruption."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("stop")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_no_not_that_one_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'no not that one' while agent is speaking triggers interruption."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("no not that one")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 1


class TestScenario3_FillerWhileAgentQuiet:
    """Test Case 3: User filler while agent quiet → registered."""

    def test_umm_while_quiet(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'umm' while agent is quiet is registered."""
        handler.set_agent_speaking(False)
        ev = create_speech_event("umm")

        handler.on_final_transcript(ev)

        # Should forward to base hooks (agent is quiet, so register all speech)
        assert len(mock_hooks.on_final_transcript_calls) == 1
        assert mock_hooks.on_final_transcript_calls[0].alternatives[0].text == "umm"

    def test_uh_while_quiet(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'uh' while agent is quiet is registered."""
        handler.set_agent_speaking(False)
        ev = create_speech_event("uh")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 1


class TestScenario4_MixedFillerAndCommand:
    """Test Case 4: Mixed filler and command → allowed (contains valid command)."""

    def test_umm_okay_stop_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'umm okay stop' while agent is speaking triggers interruption."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("umm okay stop")

        handler.on_final_transcript(ev)

        # Should forward because it contains interrupt command
        assert len(mock_hooks.on_final_transcript_calls) == 1
        assert mock_hooks.on_final_transcript_calls[0].alternatives[0].text == "umm okay stop"

    def test_uh_wait_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'uh wait' while agent is speaking triggers interruption."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("uh wait")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_hmm_no_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test 'hmm no' while agent is speaking triggers interruption."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("hmm no")

        handler.on_final_transcript(ev)

        assert len(mock_hooks.on_final_transcript_calls) == 1


class TestScenario5_LowConfidenceTranscript:
    """Test Case 5: Background murmur (low confidence) → ignored if under threshold."""

    def test_low_confidence_with_threshold(
        self, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test low confidence transcript is ignored when threshold is set."""
        config = FillerWordHandlerConfig(
            ignored_words=["uh", "umm"],
            interrupt_commands=["stop"],
            min_confidence_threshold=0.5,
        )
        handler = FillerWordHandler(base_hooks=mock_hooks, config=config)
        handler.set_agent_speaking(True)

        # Low confidence transcript
        ev = create_speech_event("hmm yeah", confidence=0.3)

        handler.on_final_transcript(ev)

        # Should be ignored due to low confidence
        assert len(mock_hooks.on_final_transcript_calls) == 0

    def test_high_confidence_with_threshold(
        self, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test high confidence transcript passes through threshold."""
        config = FillerWordHandlerConfig(
            ignored_words=["uh", "umm"],
            interrupt_commands=["stop"],
            min_confidence_threshold=0.5,
        )
        handler = FillerWordHandler(base_hooks=mock_hooks, config=config)
        handler.set_agent_speaking(True)

        # High confidence transcript (even if it's a filler)
        ev = create_speech_event("hmm yeah", confidence=0.8)

        handler.on_final_transcript(ev)

        # Should pass through (not a filler-only, so not ignored)
        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_zero_threshold_allows_all(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test that zero threshold doesn't filter by confidence."""
        handler.set_agent_speaking(True)

        ev = create_speech_event("hello", confidence=0.1)

        handler.on_final_transcript(ev)

        # Should pass through (not a filler)
        assert len(mock_hooks.on_final_transcript_calls) == 1


class TestInterimTranscripts:
    """Test interim transcript handling."""

    def test_interim_filler_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test interim filler transcript while agent is speaking."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("uh", is_final=False)

        handler.on_interim_transcript(ev, speaking=True)

        # Interim transcripts with fillers should be filtered
        # The actual filtering happens in on_final_transcript, but interim
        # may still be forwarded for display purposes, just won't trigger interruption
        # For now, we check it's not filtered in on_interim_transcript
        # (interruptions are triggered by final transcripts)
        assert len(mock_hooks.on_interim_transcript_calls) == 0

    def test_interim_valid_while_speaking(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test interim valid transcript while agent is speaking."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("wait", is_final=False)

        handler.on_interim_transcript(ev, speaking=True)

        # Valid transcripts should be forwarded
        assert len(mock_hooks.on_interim_transcript_calls) == 1


class TestStateManagement:
    """Test agent speaking state management."""

    def test_state_transition(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test that state transitions work correctly."""
        # Agent starts speaking
        handler.set_agent_speaking(True)

        # Filler should be ignored
        ev1 = create_speech_event("uh")
        handler.on_final_transcript(ev1)
        assert len(mock_hooks.on_final_transcript_calls) == 0

        # Agent stops speaking
        handler.set_agent_speaking(False)

        # Same filler should be registered
        ev2 = create_speech_event("uh")
        handler.on_final_transcript(ev2)
        assert len(mock_hooks.on_final_transcript_calls) == 1


class TestConfiguration:
    """Test configuration and dynamic updates."""

    def test_custom_ignored_words(
        self, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test custom ignored words configuration."""
        config = FillerWordHandlerConfig(
            ignored_words=["xyz", "abc"],
            interrupt_commands=["stop"],
        )
        handler = FillerWordHandler(base_hooks=mock_hooks, config=config)
        handler.set_agent_speaking(True)

        # Custom filler should be ignored
        ev = create_speech_event("xyz")
        handler.on_final_transcript(ev)
        assert len(mock_hooks.on_final_transcript_calls) == 0

        # Default filler should pass through
        ev2 = create_speech_event("uh")
        handler.on_final_transcript(ev2)
        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_custom_interrupt_commands(
        self, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test custom interrupt commands configuration."""
        config = FillerWordHandlerConfig(
            ignored_words=["uh"],
            interrupt_commands=["halt", "pause"],
        )
        handler = FillerWordHandler(base_hooks=mock_hooks, config=config)
        handler.set_agent_speaking(True)

        # Custom interrupt command should work
        ev = create_speech_event("halt")
        handler.on_final_transcript(ev)
        assert len(mock_hooks.on_final_transcript_calls) == 1

        # Default interrupt command should be treated as regular text
        ev2 = create_speech_event("stop")
        handler.on_final_transcript(ev2)
        # "stop" is not a filler, so it passes through
        assert len(mock_hooks.on_final_transcript_calls) == 2

    def test_dynamic_update(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test dynamic configuration updates."""
        handler.set_agent_speaking(True)

        # Initially "test" is not a filler
        ev1 = create_speech_event("test")
        handler.on_final_transcript(ev1)
        assert len(mock_hooks.on_final_transcript_calls) == 1

        # Update config to add "test" as filler
        handler.update_config(ignored_words=["uh", "umm", "test"])

        # Now "test" should be filtered
        ev2 = create_speech_event("test")
        handler.on_final_transcript(ev2)
        # Should still be 1 (not forwarded)
        assert len(mock_hooks.on_final_transcript_calls) == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_transcript(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test empty transcript handling."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("")

        handler.on_final_transcript(ev)

        # Empty transcripts should pass through (not filler-only)
        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_whitespace_only(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test whitespace-only transcript."""
        handler.set_agent_speaking(True)
        ev = create_speech_event("   ")

        handler.on_final_transcript(ev)

        # Whitespace-only should pass through (not filler-only)
        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_case_sensitivity(
        self, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test case sensitivity configuration."""
        config = FillerWordHandlerConfig(
            ignored_words=["uh"],
            interrupt_commands=["stop"],
            case_sensitive=True,
        )
        handler = FillerWordHandler(base_hooks=mock_hooks, config=config)
        handler.set_agent_speaking(True)

        # Uppercase should not match with case_sensitive=True
        ev = create_speech_event("UH")
        handler.on_final_transcript(ev)
        assert len(mock_hooks.on_final_transcript_calls) == 1

        # Lowercase should match
        ev2 = create_speech_event("uh")
        handler.on_final_transcript(ev2)
        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_word_boundaries(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test that word boundaries prevent partial matches."""
        handler.set_agent_speaking(True)

        # "hmm" in "hammer" should not match
        ev = create_speech_event("hammer")
        handler.on_final_transcript(ev)
        assert len(mock_hooks.on_final_transcript_calls) == 1

    def test_delegate_other_hooks(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test that other hooks are properly delegated."""
        from livekit.agents import vad
        from livekit.agents.vad import VADEvent, VADEventType

        vad_ev = VADEvent(
            type=VADEventType.START_OF_SPEECH,
            samples_index=0,
            timestamp=0.0,
            speech_duration=0.0,
            silence_duration=0.0,
        )

        handler.on_start_of_speech(vad_ev)
        handler.on_vad_inference_done(vad_ev)
        handler.on_end_of_speech(vad_ev)
        handler.on_end_of_turn(MagicMock())
        handler.on_preemptive_generation(MagicMock())
        handler.retrieve_chat_ctx()

        assert len(mock_hooks.on_start_of_speech_calls) == 1
        assert len(mock_hooks.on_vad_inference_done_calls) == 1
        assert len(mock_hooks.on_end_of_speech_calls) == 1
        assert len(mock_hooks.on_end_of_turn_calls) == 1
        assert len(mock_hooks.on_preemptive_generation_calls) == 1
        assert mock_hooks.retrieve_chat_ctx_calls == 1


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_rapid_state_changes(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test rapid state changes don't cause issues."""
        # Rapidly toggle state
        handler.set_agent_speaking(True)
        handler.set_agent_speaking(False)
        handler.set_agent_speaking(True)

        # Filler should be ignored (agent is speaking)
        ev = create_speech_event("uh")
        handler.on_final_transcript(ev)
        assert len(mock_hooks.on_final_transcript_calls) == 0

    def test_filler_with_punctuation(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test filler words with punctuation."""
        handler.set_agent_speaking(True)

        # Filler with punctuation should still be detected
        ev = create_speech_event("uh.")
        handler.on_final_transcript(ev)
        # Should be filtered (contains filler)
        assert len(mock_hooks.on_final_transcript_calls) == 0

    def test_interrupt_command_with_filler_prefix(
        self, handler: FillerWordHandler, mock_hooks: MockRecognitionHooks
    ) -> None:
        """Test interrupt command preceded by filler."""
        handler.set_agent_speaking(True)

        ev = create_speech_event("uh stop")
        handler.on_final_transcript(ev)

        # Should allow interruption (contains command)
        assert len(mock_hooks.on_final_transcript_calls) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

