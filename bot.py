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
import json
import random

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
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.heygen.api import AvatarQuality, NewSessionRequest
from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

logger.info("Loading Pipecat Flows...")
from pipecat_flows import FlowManager, NodeConfig, FlowsFunctionSchema, FlowArgs
from datetime import datetime
from typing import Optional
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logger.warning("python-dateutil not available. Date parsing will be limited.")

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


# ============================================================================
# Helper Functions for Progress Tracking
# ============================================================================

def get_collection_progress(flow_manager: FlowManager) -> dict:
    """Get the current progress of data collection.
    
    Returns:
        Dictionary with status of each required field
    """
    return {
        "name_collected": "first_name" in flow_manager.state and "surname" in flow_manager.state,
        "dob_collected": "date_of_birth" in flow_manager.state,
        "preference_collected": "reading_type" in flow_manager.state,
    }


def get_progress_message(flow_manager: FlowManager) -> str:
    """Generate a progress message for the user.
    
    Returns:
        String describing what information has been collected
    """
    progress = get_collection_progress(flow_manager)
    collected = []
    if progress["name_collected"]:
        collected.append("name")
    if progress["dob_collected"]:
        collected.append("date of birth")
    if progress["preference_collected"]:
        collected.append("reading preference")
    
    if not collected:
        return "We haven't collected any information yet."
    return f"Great! I have your {', '.join(collected)}."


# ============================================================================
# Tarot Card Loading Utilities
# ============================================================================

def load_tarot_cards() -> list:
    """Load tarot cards from JSON file.
    
    Returns:
        List of card dictionaries with name, upright_core, reversed_core, and vibe
    """
    # Get the path to the tarot card JSON file
    # Assuming the file is in spiritual_data/ relative to the bot.py location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "..", "spiritual_data", "tarot_card_meanings.json")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cards = json.load(f)
        logger.info(f"Loaded {len(cards)} tarot cards from {json_path}")
        return cards
    except FileNotFoundError:
        logger.error(f"Tarot card file not found at {json_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing tarot card JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading tarot cards: {e}")
        return []


def pull_three_cards() -> list:
    """Randomly select 3 cards from the tarot deck.
    
    Returns:
        List of 3 card dictionaries
    """
    cards = load_tarot_cards()
    if len(cards) < 3:
        logger.warning(f"Not enough cards in deck ({len(cards)}). Need at least 3.")
        return cards[:3] if cards else []
    
    selected = random.sample(cards, 3)
    logger.info(f"Selected 3 cards: {[card['name'] for card in selected]}")
    return selected


# ============================================================================
# Function Handlers for Data Collection (with Validation)
# ============================================================================

async def record_name_and_surname(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, Optional[NodeConfig]]:
    """Record the user's first name and surname with validation.
    
    Args:
        args: Contains 'first_name' and 'surname' keys
        flow_manager: FlowManager instance for state management
        
    Returns:
        Tuple of (result message, next node or None)
    """
    first_name = args.get("first_name", "").strip()
    surname = args.get("surname", "").strip()
    
    # Validation: Check if names are provided and reasonable length
    if not first_name or len(first_name) < 1:
        return "I'm sorry, I didn't catch your first name clearly. Could you tell me your first name again?", None
    
    if not surname or len(surname) < 1:
        return "I'm sorry, I didn't catch your surname clearly. Could you tell me your last name again?", None
    
    # Additional validation: Check for obviously invalid inputs
    if len(first_name) > 50 or len(surname) > 50:
        return "That name seems unusually long. Could you repeat your first and last name for me?", None
    
    # Save to state
    flow_manager.state["first_name"] = first_name
    flow_manager.state["surname"] = surname
    flow_manager.state["full_name"] = f"{first_name} {surname}".strip()
    
    logger.info(f"Recorded name: {flow_manager.state['full_name']}")
    
    # Return acknowledgment message that will be spoken, then transition
    acknowledgment = f"Thank you, {first_name}. I've noted your name. Now, could you tell me your date of birth?"
    
    # Transition to date of birth collection node
    return acknowledgment, create_collect_dob_node()


async def record_date_of_birth(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, Optional[NodeConfig]]:
    """Record the user's date of birth with validation and parsing.
    
    Args:
        args: Contains 'date_of_birth' key (format: YYYY-MM-DD or natural language)
        flow_manager: FlowManager instance for state management
        
    Returns:
        Tuple of (result message, next node or None)
    """
    dob_raw = args.get("date_of_birth", "").strip()
    
    # Validation: Check if date is provided
    if not dob_raw:
        return "I'm sorry, I didn't catch that date clearly. Could you tell me your date of birth again?", None
    
    # Try to parse the date (Recommendation 4: Date Parsing)
    dob_parsed = None
    dob_iso = None
    
    if DATEUTIL_AVAILABLE:
        try:
            dob_parsed = date_parser.parse(dob_raw)
            dob_iso = dob_parsed.isoformat()
            # Validate reasonable date range (not in the future, not too old)
            current_year = datetime.now().year
            if dob_parsed.year > current_year:
                return "That date seems to be in the future. Could you tell me your date of birth again?", None
            if dob_parsed.year < 1900:
                return "That date seems quite old. Could you confirm your date of birth?", None
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse date: {dob_raw}, error: {e}")
            # Continue with raw date if parsing fails
    
    # Save to state (both raw and parsed - Recommendation 4)
    flow_manager.state["date_of_birth"] = dob_raw
    if dob_iso:
        flow_manager.state["date_of_birth_iso"] = dob_iso
        flow_manager.state["date_of_birth_parsed"] = dob_parsed.isoformat()
        logger.info(f"Recorded date of birth: {dob_raw} (parsed: {dob_iso})")
    else:
        logger.info(f"Recorded date of birth (raw only): {dob_raw}")
    
    # Return acknowledgment message that will be spoken, then transition
    acknowledgment = f"Perfect! I've noted your date of birth as {dob_raw}. Now, what type of reading are you interested in? I offer tarot, numerology, or astrology readings."
    
    # Transition to reading preference collection node
    return acknowledgment, create_collect_preference_node()


async def record_reading_preference(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, Optional[NodeConfig]]:
    """Record the user's reading preference (without confirmation yet).
    
    Args:
        args: Contains 'reading_type' key (tarot, numerology, or astrology)
        flow_manager: FlowManager instance for state management
        
    Returns:
        Tuple of (result message, next node for confirmation)
    """
    reading_type = args.get("reading_type", "").strip().lower()
    
    # Normalize reading type
    if "tarot" in reading_type:
        reading_type = "tarot"
    elif "numerology" in reading_type:
        reading_type = "numerology"
    elif "astrology" in reading_type or "astrological" in reading_type:
        reading_type = "astrology"
    else:
        # If unclear, ask for clarification
        return "I want to make sure I understand correctly. Are you interested in a tarot, numerology, or astrology reading?", None
    
    # Save to state temporarily (will be confirmed in confirmation node)
    flow_manager.state["reading_type_pending"] = reading_type
    
    logger.info(f"Recorded reading preference (pending confirmation): {reading_type}")
    
    # Return acknowledgment and ask for confirmation
    acknowledgment = f"Wonderful! So you'd like a {reading_type} reading. Is that correct?"
    
    # Transition to confirmation node (Recommendation 5: Confirmation)
    return acknowledgment, create_confirm_preference_node()


async def confirm_reading_preference(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Confirm the reading preference and route to appropriate node.
    
    Args:
        args: Contains 'confirmed' key (true/false/yes/no)
        flow_manager: FlowManager instance for state management
        
    Returns:
        Tuple of (result message, next node)
    """
    confirmed = args.get("confirmed", "").strip().lower()
    reading_type = flow_manager.state.get("reading_type_pending", "")
    
    # Check if user confirmed
    if confirmed in ["no", "nope", "false", "incorrect", "wrong", "change", "different"]:
        # User wants to change - go back to preference collection
        flow_manager.state.pop("reading_type_pending", None)
        return "No problem! Let me ask again. What type of reading are you interested in: tarot, numerology, or astrology?", create_collect_preference_node()
    
    # User confirmed - proceed
    if not reading_type:
        # Fallback if reading_type_pending was lost
        reading_type = flow_manager.state.get("reading_type", "tarot")
    
    # Normalize and route (Recommendation 7: Pass flow_manager to access state)
    if reading_type == "tarot":
        next_node = create_tarot_node(flow_manager)
    elif reading_type == "numerology":
        next_node = create_numerology_node(flow_manager)
    elif reading_type == "astrology":
        next_node = create_astrology_node(flow_manager)
    else:
        # Default fallback
        reading_type = "tarot"
        next_node = create_tarot_node(flow_manager)
        logger.warning(f"Unexpected reading type, defaulting to tarot")
    
    # Save confirmed reading type
    flow_manager.state["reading_type"] = reading_type
    flow_manager.state.pop("reading_type_pending", None)
    
    logger.info(f"Confirmed reading preference: {reading_type}")
    
    # Return acknowledgment and transition to reading
    acknowledgment = f"Perfect! Let's begin your {reading_type} reading. I'm so excited to share this journey with you."
    
    # Transition to reading node
    return acknowledgment, next_node


async def pull_tarot_cards_handler(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, Optional[NodeConfig]]:
    """Pull 3 random tarot cards and store them in state.
    
    Args:
        args: Not used, but required by handler signature
        flow_manager: FlowManager instance for state management
        
    Returns:
        Tuple of (result message, next node or None to stay in current node)
    """
    cards = pull_three_cards()
    if not cards or len(cards) < 3:
        error_msg = "I'm sorry, I couldn't load the tarot cards. Please try again later."
        logger.error("Failed to pull tarot cards")
        return error_msg, None
    
    # Store cards in state
    flow_manager.state["tarot_cards"] = cards
    logger.info(f"Pulled and stored 3 tarot cards: {[card['name'] for card in cards]}")
    
    # Return None for next node to stay in current node
    return "Cards have been drawn.", None


async def handle_recovery(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, Optional[NodeConfig]]:
    """Handle recovery from errors or user requests to start over.
    
    Args:
        args: Contains 'action' key (restart, continue, or specific field to re-enter)
        flow_manager: FlowManager instance for state management
        
    Returns:
        Tuple of (result message, next node)
    """
    action = args.get("action", "").strip().lower()
    progress = get_collection_progress(flow_manager)
    
    if "restart" in action or "start over" in action or "begin again" in action:
        # Clear all state and restart
        flow_manager.state.clear()
        logger.info("User requested restart - clearing all state")
        return "Of course! Let's start fresh. I'm Luna, and I'm here to help you explore a reading.", create_greeting_node()
    
    elif "name" in action or "first name" in action or "surname" in action:
        # Re-enter name
        flow_manager.state.pop("first_name", None)
        flow_manager.state.pop("surname", None)
        flow_manager.state.pop("full_name", None)
        return "No problem! Let's get your name again. What's your first name and surname?", create_collect_name_node()
    
    elif "date" in action or "birth" in action or "dob" in action:
        # Re-enter date of birth
        flow_manager.state.pop("date_of_birth", None)
        flow_manager.state.pop("date_of_birth_iso", None)
        flow_manager.state.pop("date_of_birth_parsed", None)
        return "Of course! What's your date of birth?", create_collect_dob_node()
    
    elif "reading" in action or "preference" in action or "type" in action:
        # Re-enter reading preference
        flow_manager.state.pop("reading_type", None)
        flow_manager.state.pop("reading_type_pending", None)
        return "Sure! What type of reading are you interested in: tarot, numerology, or astrology?", create_collect_preference_node()
    
    else:
        # Continue from where we left off
        if not progress["name_collected"]:
            return "Let's continue. What's your first name and surname?", create_collect_name_node()
        elif not progress["dob_collected"]:
            return "Great! Now, what's your date of birth?", create_collect_dob_node()
        elif not progress["preference_collected"]:
            return "Perfect! What type of reading are you interested in: tarot, numerology, or astrology?", create_collect_preference_node()
        else:
            # All collected, proceed to reading (Recommendation 7: Pass flow_manager)
            reading_type = flow_manager.state.get("reading_type", "tarot")
            if reading_type == "tarot":
                return "Let's continue with your tarot reading.", create_tarot_node(flow_manager)
            elif reading_type == "numerology":
                return "Let's continue with your numerology reading.", create_numerology_node(flow_manager)
            else:
                return "Let's continue with your astrology reading.", create_astrology_node(flow_manager)


# ============================================================================
# Node Creation Functions (Sequential Flow - Recommendation 1)
# ============================================================================

async def proceed_to_name_collection(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Transition from greeting to name collection.
    
    Args:
        args: Not used, but required by handler signature
        flow_manager: FlowManager instance
        
    Returns:
        Tuple of (result message, next node)
    """
    # Return empty message - the next node will ask for the name
    return "", create_collect_name_node()


def create_greeting_node(user_name: str = None) -> NodeConfig:
    """Create the initial greeting node.
    
    This is the entry point that welcomes the user and transitions to name collection.
    
    Args:
        user_name: Optional user name for personalized greeting
        
    Returns:
        NodeConfig for greeting
    """
    # Get time of day for context-aware greeting
    current_hour = datetime.now().hour
    if current_hour < 12:
        time_greeting = "Good morning"
    elif current_hour < 17:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"
    
    # Build personalized greeting message that includes asking for name
    if user_name:
        greeting_content = f"{time_greeting}, {user_name}! I'm Luna, and I'm so grateful you've chosen to explore a reading with me today. To provide you with the most meaningful reading, I'll need to collect a few details. Let's start with your name - could you tell me your first name and surname?"
    else:
        greeting_content = f"{time_greeting}! I'm Luna, and I'm so grateful you've chosen to explore a reading with me today. To provide you with the most meaningful reading, I'll need to collect a few details. Let's start with your name - could you tell me your first name and surname?"
    
    # Name function for the greeting node (so user doesn't have to speak first)
    name_function = FlowsFunctionSchema(
        name="record_name_and_surname",
        description="MANDATORY: Call this function immediately when the user provides their first name AND surname. Extract both names from their response and call this function. Do not wait or ask for confirmation - call it as soon as you have both names.",
        required=["first_name", "surname"],
        handler=record_name_and_surname,
        properties={
            "first_name": {
                "type": "string",
                "description": "The user's first name"
            },
            "surname": {
                "type": "string",
                "description": "The user's last name or surname"
            }
        },
    )
    
    # Recovery function available globally
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over, correct information, or if there's any confusion. The user might say things like 'start over', 'that's wrong', 'change my name', etc.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart', 'change name', 'change date', 'change reading type', or 'continue'"
            }
        },
    )
    
    return NodeConfig(
        name="greeting",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice. Keep your responses brief and heartfelt, typically one to three sentences. Listen deeply to what the seeker shares and respond with genuine empathy and intuitive wisdom, making them feel truly heard and understood."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": f"""{greeting_content}

CRITICAL: When the user provides their name, you MUST immediately call the record_name_and_surname function with their first name and surname. After calling the function, the function will return an acknowledgment message - you MUST speak that acknowledgment message to the user."""
            }
        ],
        functions=[name_function, recovery_function],
        respond_immediately=True,
    )


def create_collect_name_node() -> NodeConfig:
    """Create a node specifically for collecting the user's name.
    
    Returns:
        NodeConfig for name collection
    """
    name_function = FlowsFunctionSchema(
        name="record_name_and_surname",
        description="MANDATORY: Call this function immediately when the user provides their first name AND surname. Extract both names from their response and call this function. Do not wait or ask for confirmation - call it as soon as you have both names.",
        required=["first_name", "surname"],
        handler=record_name_and_surname,
        properties={
            "first_name": {
                "type": "string",
                "description": "The user's first name"
            },
            "surname": {
                "type": "string",
                "description": "The user's last name or surname"
            }
        },
    )
    
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over, correct information, or if there's any confusion.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart', 'change name', or 'continue'"
            }
        },
    )
    
    return NodeConfig(
        name="collect_name",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice. Keep your responses brief and heartfelt, typically one to three sentences."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": """Ask the user for their first name and surname. Be warm and conversational. 

CRITICAL: When the user provides their name, you MUST immediately call the record_name_and_surname function with their first name and surname. After calling the function, the function will return an acknowledgment message - you MUST speak that acknowledgment message to the user. If the user only provides one name, ask for the other before calling the function."""
            }
        ],
        functions=[name_function, recovery_function],
        respond_immediately=True,  # Respond immediately to prompt the user
    )


def create_collect_dob_node() -> NodeConfig:
    """Create a node specifically for collecting the user's date of birth.
    
    Returns:
        NodeConfig for date of birth collection
    """
    dob_function = FlowsFunctionSchema(
        name="record_date_of_birth",
        description="MANDATORY: Call this function immediately when the user provides their date of birth. Accept dates in any format (e.g., 'January 15, 1990', '1990-01-15', '01/15/1990'). Extract the date from their response and call this function right away. Do not wait or ask for confirmation.",
        required=["date_of_birth"],
        handler=record_date_of_birth,
        properties={
            "date_of_birth": {
                "type": "string",
                "description": "The user's date of birth in any format they provide"
            }
        },
    )
    
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over, correct information, or if there's any confusion.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart', 'change date', 'change name', or 'continue'"
            }
        },
    )
    
    return NodeConfig(
        name="collect_dob",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice. Keep your responses brief and heartfelt, typically one to three sentences."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": """Now ask the user for their date of birth. Be warm and conversational. Accept any date format they provide.

CRITICAL: When the user provides their date of birth, you MUST immediately call the record_date_of_birth function with the date they provided. After calling the function, the function will return an acknowledgment message - you MUST speak that acknowledgment message to the user."""
            }
        ],
        functions=[dob_function, recovery_function],
        respond_immediately=True,
    )


def create_collect_preference_node() -> NodeConfig:
    """Create a node specifically for collecting the user's reading preference.
    
    Returns:
        NodeConfig for reading preference collection
    """
    preference_function = FlowsFunctionSchema(
        name="record_reading_preference",
        description="MANDATORY: Call this function immediately when the user indicates they want a tarot, numerology, or astrology reading. Extract their choice from their response and call this function right away. Do not wait or ask for confirmation.",
        required=["reading_type"],
        handler=record_reading_preference,
        properties={
            "reading_type": {
                "type": "string",
                "description": "The type of reading the user wants: 'tarot', 'numerology', or 'astrology'"
            }
        },
    )
    
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over, correct information, or if there's any confusion.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart', 'change reading type', 'change name', 'change date', or 'continue'"
            }
        },
    )
    
    return NodeConfig(
        name="collect_preference",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice. Keep your responses brief and heartfelt, typically one to three sentences."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": """Now ask the user what type of reading they're interested in: tarot, numerology, or astrology. Be warm and help them choose if they're unsure.

CRITICAL: When the user indicates their reading preference (tarot, numerology, or astrology), you MUST immediately call the record_reading_preference function with their choice. After calling the function, the function will return an acknowledgment message - you MUST speak that acknowledgment message to the user."""
            }
        ],
        functions=[preference_function, recovery_function],
        respond_immediately=True,
    )


def create_confirm_preference_node() -> NodeConfig:
    """Create a node for confirming the reading preference (Recommendation 5).
    
    Returns:
        NodeConfig for preference confirmation
    """
    confirm_function = FlowsFunctionSchema(
        name="confirm_reading_preference",
        description="MANDATORY: Call this function immediately when the user responds to your confirmation question. If they say yes/correct/right, use 'yes'. If they say no/wrong/change/different, use 'no' or 'change'. Extract their response and call this function right away.",
        required=["confirmed"],
        handler=confirm_reading_preference,
        properties={
            "confirmed": {
                "type": "string",
                "description": "Whether the user confirms: 'yes', 'no', 'correct', 'wrong', 'change', etc."
            }
        },
    )
    
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over or correct information.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart', 'change reading type', etc."
            }
        },
    )
    
    return NodeConfig(
        name="confirm_preference",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice. Keep your responses brief and heartfelt, typically one to three sentences."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": """Confirm the user's reading preference by repeating it back to them and asking if that's correct. For example: 'Perfect! So you'd like a tarot reading, is that right?'

CRITICAL: When the user responds (yes, no, correct, wrong, change, etc.), you MUST immediately call the confirm_reading_preference function with their response. After calling the function, the function will return an acknowledgment message - you MUST speak that acknowledgment message to the user."""
            }
        ],
        functions=[confirm_function, recovery_function],
        respond_immediately=True,
    )


def create_tarot_node(flow_manager: Optional[FlowManager] = None) -> NodeConfig:
    """Create a tarot reading node that pulls 3 cards and provides a reading.
    
    Args:
        flow_manager: Optional FlowManager to access state. If None, state will be
                     accessed via function handlers.
    
    Returns:
        NodeConfig for tarot reading flow
    """
    # Build personalized message using state if available
    user_name = None
    dob = None
    if flow_manager:
        user_name = flow_manager.state.get("full_name") or flow_manager.state.get("first_name")
        dob = flow_manager.state.get("date_of_birth")
    
    # Pull 3 random tarot cards
    cards = pull_three_cards()
    if not cards or len(cards) < 3:
        logger.error("Failed to pull 3 tarot cards")
        # Fallback message if cards can't be loaded
        task_content = "I'm sorry, I'm having trouble loading the tarot cards right now. Please try again later."
        if flow_manager:
            flow_manager.state["tarot_cards"] = []
    else:
        # Store cards in state if flow_manager is available
        if flow_manager:
            flow_manager.state["tarot_cards"] = cards
        
        # Format card data for the LLM
        card_descriptions = []
        for i, card in enumerate(cards, 1):
            card_desc = f"""Card {i}: {card['name']}
Upright meaning: {card['upright_core']}
Reversed meaning: {card['reversed_core']}
Vibe: {card['vibe']}"""
            card_descriptions.append(card_desc)
        
        cards_text = "\n\n".join(card_descriptions)
        
        # Build personalized welcome message
        welcome_msg = "Welcome to your tarot reading"
        if user_name:
            welcome_msg = f"Welcome to your tarot reading, {user_name}"
        
        # Build task content with card data and instructions
        task_content = f"""{welcome_msg}. I've drawn three cards for you. Here are the cards and their meanings:

{cards_text}

Your task is to:
1. Name and describe each of the three cards in one sentence. Use the upright or reversed meaning based on what feels most appropriate for the reading context. You may choose upright or reversed for each card independently.
2. After describing all three cards, provide a combined reading that explains how these three cards work together and what message they convey as a whole.

Speak naturally and conversationally. Your words will be spoken aloud, so avoid special characters, emojis, or bullet points. Keep each card description to one sentence, then provide a thoughtful combined reading."""
        
        if user_name and dob:
            task_content += f"\n\nYou know the user's name is {user_name} and their date of birth is {dob}. Use this information to personalize the reading if relevant."
        elif user_name:
            task_content += f"\n\nYou know the user's name is {user_name}. Use this to personalize the reading if relevant."
        elif dob:
            task_content += f"\n\nYou know the user's date of birth is {dob}. Use this information in the reading if relevant."
    
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over or go back.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart' or 'continue'"
            }
        },
    )
    
    return NodeConfig(
        name="tarot_reading",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide specializing in tarot readings. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice. You only use the meanings provided in the card data. Do not quote the card data directly; expand the ideas naturally. Avoid mystical jargon, predictions, or fate-based statements. Stay reflective, symbolic, and emotionally gentle."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": task_content
            }
        ],
        functions=[recovery_function],
    )


def create_numerology_node(flow_manager: Optional[FlowManager] = None) -> NodeConfig:
    """Create a skeleton node for numerology readings (Recommendation 7: Access State).
    
    Args:
        flow_manager: Optional FlowManager to access state. If None, state will be
                     accessed via function handlers.
    
    Returns:
        NodeConfig for numerology reading flow
    """
    # Build personalized message using state if available
    user_name = None
    dob = None
    dob_iso = None
    if flow_manager:
        user_name = flow_manager.state.get("full_name") or flow_manager.state.get("first_name")
        dob = flow_manager.state.get("date_of_birth")
        dob_iso = flow_manager.state.get("date_of_birth_iso")
    
    welcome_msg = "Welcome to your numerology reading"
    if user_name:
        welcome_msg = f"Welcome to your numerology reading, {user_name}"
    
    task_content = f"""{welcome_msg}. This is a skeleton node - you can begin the numerology reading process here."""
    
    if user_name and dob:
        task_content += f" You know the user's name is {user_name} and their date of birth is {dob}."
        if dob_iso:
            task_content += f" The parsed date is {dob_iso} which you can use for calculations."
        task_content += " Use this information to calculate their numerology numbers and provide insights."
    elif user_name:
        task_content += f" You know the user's name is {user_name}. Use this to calculate numerology numbers."
    elif dob:
        task_content += f" You know the user's date of birth is {dob}."
        if dob_iso:
            task_content += f" The parsed date is {dob_iso} which you can use for calculations."
        task_content += " Use this for numerology calculations."
    
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over or go back.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart' or 'continue'"
            }
        },
    )
    
    return NodeConfig(
        name="numerology_reading",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide specializing in numerology readings. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": task_content
            }
        ],
        functions=[recovery_function],
    )


def create_astrology_node(flow_manager: Optional[FlowManager] = None) -> NodeConfig:
    """Create a skeleton node for astrology readings (Recommendation 7: Access State).
    
    Args:
        flow_manager: Optional FlowManager to access state. If None, state will be
                     accessed via function handlers.
    
    Returns:
        NodeConfig for astrology reading flow
    """
    # Build personalized message using state if available
    user_name = None
    dob = None
    dob_iso = None
    if flow_manager:
        user_name = flow_manager.state.get("full_name") or flow_manager.state.get("first_name")
        dob = flow_manager.state.get("date_of_birth")
        dob_iso = flow_manager.state.get("date_of_birth_iso")
    
    welcome_msg = "Welcome to your astrology reading"
    if user_name:
        welcome_msg = f"Welcome to your astrology reading, {user_name}"
    
    task_content = f"""{welcome_msg}. This is a skeleton node - you can begin the astrology reading process here."""
    
    if user_name and dob:
        task_content += f" You know the user's name is {user_name} and their date of birth is {dob}."
        if dob_iso:
            task_content += f" The parsed date is {dob_iso} which you can use to determine their astrological chart."
        task_content += " Use this information to determine their astrological chart and provide insights."
    elif user_name:
        task_content += f" You know the user's name is {user_name}. Use this to personalize the reading."
    elif dob:
        task_content += f" You know the user's date of birth is {dob}."
        if dob_iso:
            task_content += f" The parsed date is {dob_iso} which you can use to determine their astrological chart."
        task_content += " Use this to determine their astrological chart."
    
    recovery_function = FlowsFunctionSchema(
        name="handle_recovery",
        description="Use this if the user wants to start over or go back.",
        required=["action"],
        handler=handle_recovery,
        properties={
            "action": {
                "type": "string",
                "description": "The action the user wants: 'restart' or 'continue'"
            }
        },
    )
    
    return NodeConfig(
        name="astrology_reading",
        role_messages=[
            {
                "role": "system",
                "content": "You are Luna, a warm and intuitive psychic guide specializing in astrology readings. Your words will be spoken aloud, so speak naturally and conversationally—avoid special characters, emojis, or bullet points that don't translate well to voice."
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": task_content
            }
        ],
        functions=[recovery_function],
    )


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

        # Check which LLM provider to use
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if llm_provider == "vllm":
            # Use hosted vLLM model
            gpu_llm_base_url = os.getenv("GPU_LLM_BASE_URL")
            gpu_llm_model = os.getenv("GPU_LLM_MODEL", "qwen2p5-1_5b")
            gpu_llm_temperature = float(os.getenv("GPU_LLM_TEMPERATURE", "0.8"))
            
            if not gpu_llm_base_url:
                raise ValueError("GPU_LLM_BASE_URL is not set in environment variables. Please add it to your .env file.")
            
            logger.info(f"Using vLLM model: {gpu_llm_model} at {gpu_llm_base_url}")
            
            # vLLM exposes OpenAI-compatible API, so we can use OpenAILLMService with custom base_url
            # Use a dummy API key (vLLM typically doesn't require auth, but OpenAI client expects it)
            # Note: vLLM server must be started with --enable-auto-tool-choice and --tool-call-parser flags
            # If you get tool choice errors, you may need to configure the vLLM server with those flags
            try:
                llm = OpenAILLMService(
                    api_key="dummy-key",  # vLLM doesn't require auth, but OpenAI client needs a key
                    model=gpu_llm_model,
                    base_url=gpu_llm_base_url,
                    temperature=gpu_llm_temperature,
                )
            except Exception as e:
                logger.error(f"Failed to initialize vLLM service: {e}")
                logger.error("Note: vLLM server must be started with --enable-auto-tool-choice and --tool-call-parser flags")
                logger.error("If you cannot modify the server, you may need to use a different LLM provider or configure the server")
                raise
        else:
            # Default to OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key or openai_key == "your_openai_api_key":
                raise ValueError("OPENAI_API_KEY is not set in environment variables. Please add it to your .env file.")
            
            logger.info("Using OpenAI model")
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

        # Initialize context for FlowManager
        messages = []
        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
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

        # Initialize FlowManager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            transport=transport,
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

            # Extract user context if available (e.g., from client metadata)
            user_name = None
            if hasattr(client, "name") and client.name:
                user_name = client.name
            elif hasattr(client, "metadata") and isinstance(client.metadata, dict):
                user_name = client.metadata.get("name") or client.metadata.get("user_name")
            
            # Store user context in FlowManager state for customization
            if user_name:
                flow_manager.state["user_name"] = user_name
                logger.info(f"User name detected: {user_name}")
            
            # Initialize the conversation with the greeting node
            greeting_node = create_greeting_node(user_name=user_name)
            await flow_manager.initialize(greeting_node)

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
