import os
import logging
import asyncio
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, RoomInputOptions
from livekit.plugins import deepgram, openai, silero

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(".env")

BOT_INSTRUCTIONS = """
You are an automated hotel booking assistant. Follow this flow exactly and output ONLY the response for the current step based on the latest user input. Do NOT generate multiple steps, assume future inputs (e.g., staff responses like 'YES' or booking details), or simulate the conversation. The system will provide the next input; you must wait for it. Output EXACTLY the specified text for the current step, with no additional text, assumptions, or creative additions. If no name is stored from previous inputs, extract the name from the latest input.

Flow:
1. Output: 'Hello ! How can I help you with your booking?'
2. Wait for guest input. If it contains a name (e.g., 'John Smith' or 'my name is John Smith'), extract the name and proceed to step 3. For generic greetings (e.g., 'hello', 'hi') or irrelevant inputs (e.g., 'check my booking'), output: 'Please provide your booking name to proceed.'
3. Output: 'Please hold while I confirm the booking with staff. Do we have a booking by name [name] on [Date] '
4. Wait for staff input, which must be 'YES' or 'NO'. If invalid, output: 'Please respond with "YES" or "NO" to confirm the booking.'
5. If 'NO', output: 'We do not have a booking under that name. Thank you, have a great day!'
6. If 'YES', output: 'Can you confirm how many rooms are booked and if breakfast is included?'
7. Wait for staff details (e.g., '2 rooms, breakfast included').
8. Output: 'Your booking for [name] includes: [details]. Thank you, have a great day!'

Examples:
- Input: 'Hello' → Output: 'Please provide your booking name to proceed.'
- Input: 'My name is John Smith. Please check my booking for fifth of May.' → Output: 'Please hold while I confirm the booking with staff.'
- Input: 'YES' → Output: 'Can you confirm how many rooms are booked and if breakfast is included?'
- Input: '2 rooms, breakfast included' → Output: 'Your booking for John Smith includes: 2 rooms, breakfast included. Thank you, have a great day!'
- Input: 'NO' → Output: 'We do not have a booking under the name 'John Smith'. Thank you, have a great day!'
- Input: 'Check my booking' → Output: 'Please provide your booking name to proceed.'

Negative Examples (DO NOT DO THIS):
- Input: 'My name is John Smith' → Do NOT output: 'Please hold while I confirm... YES Can you confirm...' or assume 'YES'.
- Input: 'YES' → Do NOT output: 'Your booking includes 1 room...' or assume booking details.
- Input: Any → Do NOT include unsolicited details like '1 standard room' or 'breakfast is not included.'

Do NOT simulate the conversation, assume inputs, or generate responses beyond the current step. Use the name from the input or stored context for the final step.
"""

async def entrypoint(ctx: JobContext):
    try:
        await ctx.connect()
        logger.info("Connected to LiveKit room")

        session = AgentSession(
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-3", language="en-US"),
            llm=openai.LLM.with_ollama(model="mistral", base_url="http://localhost:11434/v1"),  # Use mistral:latest
            tts=deepgram.TTS(model="aura-asteria-en"),
        )
        logger.info("AgentSession initialized with mistral")

        agent = Agent(instructions=BOT_INSTRUCTIONS)
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(close_on_disconnect=False)
        )
        logger.info("Agent started")

        await session.say("Hello! How can I help you with your booking?")
        logger.info("Initial greeting sent")

        # Keep session open for 5 minutes or until closed
        try:
            await asyncio.sleep(300)  # 5-minute timeout
        except asyncio.CancelledError:
            logger.info("Session cancelled")
        finally:
            await session.aclose()
            logger.info("Session closed")

    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        raise

if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))