"""
Filler Word Interruption Handler

This module provides intelligent interruption handling that distinguishes between
filler words/phrases and genuine user interruptions. It filters out filler sounds
when the agent is speaking while still allowing valid interruptions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import stt, vad
from ..log import logger

if TYPE_CHECKING:
    from .audio_recognition import RecognitionHooks

# Default filler words (case-insensitive matching)
DEFAULT_FILLER_WORDS = ["uh", "umm", "hmm", "haan", "uhh", "um", "er", "ah"]

# Default interruption commands that should always interrupt
DEFAULT_INTERRUPT_COMMANDS = [
    "wait",
    "stop",
    "hold on",
    "stop it",
    "cancel",
    "no",
    "not that",
    "wrong",
    "nevermind",
]


@dataclass
class FillerWordHandlerConfig:
    """Configuration for filler word filtering."""

    ignored_words: list[str] = field(
        default_factory=lambda: DEFAULT_FILLER_WORDS.copy()
    )
    """List of filler words/phrases to ignore when agent is speaking."""

    interrupt_commands: list[str] = field(
        default_factory=lambda: DEFAULT_INTERRUPT_COMMANDS.copy()
    )
    """List of commands that should always trigger interruption."""

    min_confidence_threshold: float = 0.0
    """Minimum ASR confidence threshold. Transcripts below this are ignored."""

    case_sensitive: bool = False
    """Whether word matching should be case-sensitive."""

    # Optional: Support for dynamic updates
    _dynamic_update_lock: bool = False
    """Internal lock to prevent concurrent updates."""


class FillerWordHandler:
    """
    Handles filler word filtering for intelligent interruption detection.

    This handler wraps RecognitionHooks to filter out filler-only transcripts
    when the agent is speaking, while allowing genuine interruptions through.
    """

    def __init__(
        self,
        base_hooks: RecognitionHooks,
        config: FillerWordHandlerConfig | None = None,
    ) -> None:
        """
        Initialize the filler word handler.

        Args:
            base_hooks: The underlying RecognitionHooks implementation to wrap.
            config: Configuration for filler word filtering. If None, uses defaults.
        """
        self._base_hooks = base_hooks
        self._config = config or FillerWordHandlerConfig()
        self._agent_speaking = False

        # Load configuration from environment if available
        self._load_env_config()

        # Compile regex patterns for efficient matching
        self._ignored_patterns = self._compile_patterns(self._config.ignored_words)
        self._interrupt_patterns = self._compile_patterns(
            self._config.interrupt_commands
        )

        logger.info(
            "FillerWordHandler initialized",
            extra={
                "ignored_words_count": len(self._config.ignored_words),
                "interrupt_commands_count": len(self._config.interrupt_commands),
                "min_confidence": self._config.min_confidence_threshold,
            },
        )

    def _load_env_config(self) -> None:
        """Load configuration from environment variables."""
        env_ignored = os.getenv("LIVEKIT_IGNORED_FILLER_WORDS")
        if env_ignored:
            words = [w.strip() for w in env_ignored.split(",") if w.strip()]
            if words:
                self._config.ignored_words = words
                logger.info("Loaded ignored words from environment", extra={"words": words})

        env_commands = os.getenv("LIVEKIT_INTERRUPT_COMMANDS")
        if env_commands:
            commands = [c.strip() for c in env_commands.split(",") if c.strip()]
            if commands:
                self._config.interrupt_commands = commands
                logger.info(
                    "Loaded interrupt commands from environment", extra={"commands": commands}
                )

        env_threshold = os.getenv("LIVEKIT_MIN_CONFIDENCE_THRESHOLD")
        if env_threshold:
            try:
                self._config.min_confidence_threshold = float(env_threshold)
            except ValueError:
                logger.warning(f"Invalid confidence threshold: {env_threshold}")

    def _compile_patterns(self, words: list[str]) -> list[re.Pattern]:
        """Compile word patterns for efficient matching."""
        flags = 0 if self._config.case_sensitive else re.IGNORECASE
        patterns = []
        for word in words:
            # Use word boundaries to avoid partial matches
            # Escape special regex characters
            escaped = re.escape(word)
            pattern = re.compile(rf"\b{escaped}\b", flags)
            patterns.append(pattern)
        return patterns

    def update_config(
        self,
        *,
        ignored_words: list[str] | None = None,
        interrupt_commands: list[str] | None = None,
        min_confidence_threshold: float | None = None,
    ) -> None:
        """
        Dynamically update the handler configuration.

        Args:
            ignored_words: New list of filler words to ignore.
            interrupt_commands: New list of interrupt commands.
            min_confidence_threshold: New minimum confidence threshold.
        """
        if self._config._dynamic_update_lock:
            logger.warning("Config update already in progress, skipping")
            return

        self._config._dynamic_update_lock = True
        try:
            if ignored_words is not None:
                self._config.ignored_words = ignored_words
                self._ignored_patterns = self._compile_patterns(ignored_words)
                logger.info("Updated ignored words", extra={"words": ignored_words})

            if interrupt_commands is not None:
                self._config.interrupt_commands = interrupt_commands
                self._interrupt_patterns = self._compile_patterns(interrupt_commands)
                logger.info(
                    "Updated interrupt commands", extra={"commands": interrupt_commands}
                )

            if min_confidence_threshold is not None:
                self._config.min_confidence_threshold = min_confidence_threshold
                logger.info(
                    "Updated confidence threshold",
                    extra={"threshold": min_confidence_threshold},
                )
        finally:
            self._config._dynamic_update_lock = False

    def _is_filler_only(self, text: str) -> bool:
        """
        Check if text contains only filler words.

        Args:
            text: The transcript text to check.

        Returns:
            True if text contains only filler words (after removing them).
        """
        if not text:
            return False

        # Remove filler words and check if anything meaningful remains
        cleaned = text
        for pattern in self._ignored_patterns:
            cleaned = pattern.sub("", cleaned)

        # Remove extra whitespace and check if anything is left
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return len(cleaned) == 0

    def _contains_interrupt_command(self, text: str) -> bool:
        """
        Check if text contains any interrupt command.

        Args:
            text: The transcript text to check.

        Returns:
            True if text contains an interrupt command.
        """
        if not text:
            return False

        for pattern in self._interrupt_patterns:
            if pattern.search(text):
                return True
        return False

    def _should_ignore_transcript(
        self, transcript: str, confidence: float | None = None
    ) -> bool:
        """
        Determine if a transcript should be ignored.

        Args:
            transcript: The transcript text.
            confidence: ASR confidence score (optional).

        Returns:
            True if the transcript should be ignored.
        """
        # Check confidence threshold
        if confidence is not None and confidence < self._config.min_confidence_threshold:
            logger.debug(
                "Ignoring low-confidence transcript",
                extra={"confidence": confidence, "threshold": self._config.min_confidence_threshold},
            )
            return True

        # If agent is not speaking, never ignore
        if not self._agent_speaking:
            return False

        # Always allow interrupt commands
        if self._contains_interrupt_command(transcript):
            logger.info(
                "Interrupt command detected, allowing interruption",
                extra={"transcript": transcript},
            )
            return False

        # Ignore filler-only transcripts when agent is speaking
        if self._is_filler_only(transcript):
            logger.debug(
                "Ignoring filler-only transcript while agent is speaking",
                extra={"transcript": transcript},
            )
            return True

        return False

    def set_agent_speaking(self, speaking: bool) -> None:
        """
        Update the agent speaking state.

        Args:
            speaking: True if agent is currently speaking.
        """
        was_speaking = self._agent_speaking
        self._agent_speaking = speaking

        if was_speaking != speaking:
            logger.debug(
                "Agent speaking state changed",
                extra={"speaking": speaking, "was_speaking": was_speaking},
            )

    # Delegate all RecognitionHooks methods, intercepting where needed

    def on_start_of_speech(self, ev: vad.VADEvent | None) -> None:
        """Handle start of speech event."""
        self._base_hooks.on_start_of_speech(ev)

    def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        """Handle VAD inference done event."""
        self._base_hooks.on_vad_inference_done(ev)

    def on_end_of_speech(self, ev: vad.VADEvent | None) -> None:
        """Handle end of speech event."""
        self._base_hooks.on_end_of_speech(ev)

    def on_interim_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None) -> None:
        """
        Handle interim transcript with filler word filtering.

        Intercepts transcripts and filters out filler-only segments when
        agent is speaking.
        """
        transcript = ev.alternatives[0].text if ev.alternatives else ""
        confidence = ev.alternatives[0].confidence if ev.alternatives else None

        # Filter filler words only when agent is speaking
        if self._should_ignore_transcript(transcript, confidence):
            logger.debug(
                "Filtered interim transcript",
                extra={
                    "transcript": transcript,
                    "agent_speaking": self._agent_speaking,
                    "confidence": confidence,
                },
            )
            # Still forward to base hooks, but mark as filtered
            # The interruption logic will be handled in on_final_transcript
            return

        # Forward to base hooks
        self._base_hooks.on_interim_transcript(ev, speaking=speaking)

    def on_final_transcript(self, ev: stt.SpeechEvent) -> None:
        """
        Handle final transcript with filler word filtering.

        This is where interruptions are actually triggered, so we filter
        here to prevent false interruptions.
        """
        transcript = ev.alternatives[0].text if ev.alternatives else ""
        confidence = ev.alternatives[0].confidence if ev.alternatives else None

        # Filter filler words only when agent is speaking
        if self._should_ignore_transcript(transcript, confidence):
            logger.info(
                "Filtered final transcript (filler word ignored)",
                extra={
                    "transcript": transcript,
                    "agent_speaking": self._agent_speaking,
                    "confidence": confidence,
                },
            )
            # Don't forward to base hooks - this prevents the interruption
            return

        # Log valid interruption when agent is speaking
        if self._agent_speaking:
            logger.info(
                "Valid interruption detected",
                extra={
                    "transcript": transcript,
                    "confidence": confidence,
                    "contains_interrupt_command": self._contains_interrupt_command(transcript),
                },
            )

        # Forward to base hooks
        self._base_hooks.on_final_transcript(ev)

    def on_end_of_turn(self, info) -> bool:
        """Handle end of turn event."""
        return self._base_hooks.on_end_of_turn(info)

    def on_preemptive_generation(self, info) -> None:
        """Handle preemptive generation event."""
        self._base_hooks.on_preemptive_generation(info)

    def retrieve_chat_ctx(self):
        """Retrieve chat context."""
        return self._base_hooks.retrieve_chat_ctx()

