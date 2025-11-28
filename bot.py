#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Azure Speech Services (Text-to-Speech)
- HeyGen (Video Avatar)

Run the bot using::

    uv run bot.py
"""

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

# Note: You may see Objective-C duplicate class warnings on macOS when both
# livekit (HeyGen) and daily libraries are present. These are harmless runtime
# warnings caused by both libraries bundling WebRTC. We use webrtc transport
# by default when HeyGen is configured to minimize conflicts.

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TextFrame, TranscriptionFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.heygen.api import AvatarQuality, NewSessionRequest
from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    
    async with aiohttp.ClientSession() as session:
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_key or deepgram_key == "your_deepgram_api_key":
            raise ValueError("DEEPGRAM_API_KEY is not set in environment variables. Please add it to your .env file.")
        
        deepgram_model = os.getenv("DEEPGRAM_MODEL")
        stt_kwargs = {"api_key": deepgram_key}
        if deepgram_model:
            stt_kwargs["model"] = deepgram_model
        
        stt = DeepgramSTTService(**stt_kwargs)

        azure_key = os.getenv("AZURE_SPEECH_KEY")
        azure_region = os.getenv("AZURE_SPEECH_REGION")
        if not azure_key or azure_key == "your_azure_speech_key":
            raise ValueError("AZURE_SPEECH_KEY is not set in environment variables. Please add it to your .env file.")
        if not azure_region or azure_region == "your_azure_region":
            raise ValueError("AZURE_SPEECH_REGION is not set in environment variables. Please add it to your .env file.")
        
        tts = AzureTTSService(
            api_key=azure_key,
            region=azure_region,
            voice_name=os.getenv("AZURE_TTS_VOICE", "en-US-NancyMultilingualNeural"),
        )

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_openai_api_key":
            raise ValueError("OPENAI_API_KEY is not set in environment variables. Please add it to your .env file.")
        
        llm = OpenAILLMService(api_key=openai_key)

        # HeyGen video service - COMMENTED OUT FOR NOW
        # heygen_key = os.getenv("HEYGEN_API_KEY")
        # if not heygen_key or heygen_key == "your_heygen_api_key":
        #     raise ValueError("HEYGEN_API_KEY is not set in environment variables. Please add it to your .env file.")
        # 
        # # Validate API key format (HeyGen API keys are typically longer)
        # if len(heygen_key) < 20:
        #     logger.warning(f"HEYGEN_API_KEY seems too short ({len(heygen_key)} chars). Please verify it's correct.")
        # 
        # # Configure HeyGen session with custom avatar
        # # See: https://reference-server.pipecat.ai/en/latest/api/pipecat.services.heygen.api.html#pipecat.services.heygen.api.NewSessionRequest
        # heygen_avatar_id = os.getenv("HEYGEN_AVATAR_ID", "cfd04a6141eb43dfb05faac440fa0852")
        # heygen_quality = os.getenv("HEYGEN_QUALITY", "high")  # low, medium, or high
        # 
        # logger.info(f"Creating HeyGen session with avatar_id={heygen_avatar_id}, quality={heygen_quality}")
        # 
        # # Create session request with avatar configuration
        # session_request = NewSessionRequest(
        #     avatar_id=heygen_avatar_id,
        #     quality=AvatarQuality(heygen_quality) if heygen_quality in ["low", "medium", "high"] else AvatarQuality.high,
        #     version="v2",
        # )
        # 
        # try:
        #     heygen = HeyGenVideoService(
        #         api_key=heygen_key,
        #         session=session,
        #         session_request=session_request,
        #     )
        #     logger.info("HeyGen video service initialized successfully")
        # except Exception as e:
        #     logger.error(f"Failed to initialize HeyGen service: {e}")
        #     logger.error("This might be due to:")
        #     logger.error("1. Invalid HEYGEN_API_KEY")
        #     logger.error("2. Invalid avatar_id (avatar not found or not accessible)")
        #     logger.error("3. Network/API connectivity issues")
        #     raise

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Be succinct and respond to what the user said in a creative and helpful way.",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        # Simple processor to log text messages (web UI should display them automatically)
        class TextLogger(FrameProcessor):
            async def process_frame(self, frame, direction):
                if isinstance(frame, TranscriptionFrame):
                    logger.info(f"👤 User said: {frame.text}")
                elif isinstance(frame, TextFrame) and hasattr(frame, 'text'):
                    logger.info(f"🤖 Bot said: {frame.text}")
                # Always forward the frame
                await self.push_frame(frame, direction)

        text_logger = TextLogger()

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                text_logger,  # Log and display user transcriptions
                context_aggregator.user(),  # User responses
                llm,  # LLM
                text_logger,  # Log and display bot responses
                tts,  # TTS
                # heygen,  # HeyGen avatar video - COMMENTED OUT
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=getattr(runner_args, "pipeline_idle_timeout_secs", None),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            
            # Updating publishing settings to enable adaptive bitrate for Daily transport
            try:
                from pipecat.transports.daily.transport import DailyTransport
                if isinstance(transport, DailyTransport):
                    await transport.update_publishing(
                        publishing_settings={
                            "camera": {
                                "sendSettings": {
                                    "allowAdaptiveLayers": True,
                                }
                            }
                        }
                    )
            except ImportError:
                pass  # Daily transport not available

            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": "Start by saying 'Hello' and then a short greeting.",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    
    # Lazy import Daily transport only if explicitly requested
    # This avoids loading Daily's WebRTC implementation when using HeyGen
    if hasattr(runner_args, "transport") and runner_args.transport == "daily":
        try:
            from pipecat.transports.daily.transport import DailyParams
            transport_params["daily"] = lambda: DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_is_live=True,
                video_out_width=1280,
                video_out_height=720,
                video_out_bitrate=2_000_000,  # 2MBps
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
                turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
            )
        except ImportError:
            logger.warning("Daily transport requested but not available")

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
