"""
Example: Filler Word Interruption Handler

This example demonstrates the filler word interruption handler feature that
intelligently distinguishes between filler words (like "uh", "umm", "hmm")
and genuine user interruptions.

The handler:
- Ignores filler-only transcripts when the agent is speaking
- Allows valid interruptions (commands like "wait", "stop") even with fillers
- Registers all transcripts when the agent is quiet
"""

import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    cli,
)
from livekit.agents.voice.filler_word_handler import (
    FillerWordHandlerConfig,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("filler-word-example")

load_dotenv()


class FillerWordExampleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a friendly assistant. Speak naturally and help users. "
            "When interrupted, stop immediately. Keep responses concise.",
        )

    async def on_enter(self):
        # Generate initial greeting
        self.session.generate_reply()


def prewarm(proc):
    """Preload VAD model for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()


@cli.entrypoint
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Configure filler word handler
    # Customize ignored words and interrupt commands as needed
    filler_config = FillerWordHandlerConfig(
        ignored_words=["uh", "umm", "hmm", "haan", "uhh", "um", "er", "ah"],
        interrupt_commands=[
            "wait",
            "stop",
            "hold on",
            "stop it",
            "cancel",
            "no",
            "not that",
            "wrong",
            "nevermind",
        ],
        min_confidence_threshold=0.0,  # Adjust based on your ASR model
        case_sensitive=False,
    )

    # Create session with filler word handler enabled
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        # Enable filler word handler
        filler_word_handler_config=filler_config,
    )

    # Log events for debugging
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        logger.info(f"Agent state: {ev.old_state} -> {ev.new_state}")

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        logger.info(f"User state: {ev.old_state} -> {ev.new_state}")

    await session.start(
        agent=FillerWordExampleAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(prewarm=prewarm)

