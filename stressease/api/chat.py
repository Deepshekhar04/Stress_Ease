"""
Chat API endpoints with LangChain integration.

This module provides:
- POST /message - Send chat message with implicit session creation
- POST /end-session - End chat session and cleanup resources
- GET /crisis-resources - Get country-specific crisis resources
"""

from flask import Blueprint, request, jsonify
from stressease.services.utility.auth_service import token_required
from stressease.services.chat import llm_service
from stressease.services.chat import chat_memory_service
from stressease.services.mood import mood_service
from stressease.services.chat import crisis_resource_service
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import uuid
import threading


# Create the chat blueprint
chat_bp = Blueprint("chat", __name__)


# ============================================================================
# IN-MEMORY SESSION CACHE
# ============================================================================
# Format: {user_id: {session_id: {'chain': runnable, 'last_activity': timestamp, 'message_count': int}}}
active_chat_sessions = {}


# ============================================================================
# CRISIS SUPPORT ENDPOINTS
# ============================================================================


@chat_bp.route("/crisis-resources", methods=["GET", "POST"])
@token_required
def get_crisis_resources(user_id):
    """
    Get country-specific crisis resources with intelligent caching.

    NEW: Uses SerpApi + LLM for real-time web search when cache is stale.
    - Cache TTL: 30 days
    - Returns exactly 5 contacts (1 emergency + 4 mental health)
    - Graceful fallback to cache if search fails

    Query Parameters:
        country (str): Country name selected from dropdown

    Returns:
        JSON response with country-specific crisis resources
    """
    try:
        from stressease.services.sos import get_sos_contacts

        # Get country from query parameter
        country = request.args.get("country", "").strip()

        # Default to India if not provided
        if not country:
            country = "India"

        # Get SOS contacts using intelligent service
        # This will:
        # 1. Return cached data if fresh (< 30 days)
        # 2. Fetch fresh data if stale (>= 30 days)
        # 3. Fallback to cache if fetch fails
        resources = get_sos_contacts(country)

        if not resources:
            # Ultimate fallback - try old LLM method
            print(f"⚠ SOS service failed for {country}, trying fallback LLM method")
            resources = llm_service.find_crisis_resources(country)

            if resources:
                # Cache the LLM-generated resources
                crisis_resource_service.cache_crisis_resources(country, resources)

        if not resources:
            response = jsonify(
                {
                    "success": False,
                    "message": f"Could not find crisis resources for {country}",
                }
            )
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            return response, 404

        # Determine source for transparency
        cached_at = resources.get("cached_at")
        source_type = "cache" if cached_at else "generated"

        # Return the resources
        response = jsonify(
            {
                "success": True,
                "message": f"Crisis resources retrieved successfully",
                "resources": resources,
                "source": source_type,
            }
        )
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response, 200

    except Exception as e:
        print(f"❌ Error in get_crisis_resources: {str(e)}")
        response = jsonify(
            {
                "success": False,
                "message": f"Error retrieving crisis resources: {str(e)}",
            }
        )
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response, 500


# ============================================================================
# CHAT MESSAGE ENDPOINT
# ============================================================================


@chat_bp.route("/message", methods=["POST"])
@token_required
def send_chat_message(user_id):
    """
    Send a chat message with implicit session creation and LangChain integration.

    Expected JSON payload:
    {
        "message": "Hello, I'm feeling anxious today",
        "session_id": null  // null for new session, or existing session_id
    }

    Returns:
        JSON response with AI reply and session_id
    """
    try:
        # Get and validate JSON data
        message_data = request.get_json()
        if not message_data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid request",
                        "message": "JSON data is required",
                    }
                ),
                400,
            )

        # Extract and validate message
        user_message = message_data.get("message", "").strip()
        session_id = message_data.get("session_id")

        # Input validation
        if not user_message:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid message",
                        "message": "Message cannot be empty",
                    }
                ),
                400,
            )

        # Block gibberish (messages without any letters)
        # TEMPORARILY DISABLED FOR TESTING - See how LLM handles gibberish
        # if not any(c.isalpha() for c in user_message):
        #     return (
        #         jsonify(
        #             {
        #                 "success": False,
        #                 "error": "Invalid message",
        #                 "message": "Please send a message with words so I can help you.",
        #             }
        #         ),
        #         400,
        #     )

        if len(user_message) > 1000:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Message too long",
                        "message": "Message must be 1000 characters or less",
                    }
                ),
                400,
            )

        # Get session data (chain and history)
        session_id, chain, history_messages = _load_session(session_id, user_id)

        if not chain:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Session error",
                        "message": "Could not initialize chat session",
                    }
                ),
                500,
            )

        # Generate AI response using LCEL chain
        # Pass history explicitly as it's stateless
        ai_response = llm_service.generate_chat_response(
            chain, user_message, history_messages
        )

        # Calculate new turn number
        turn_number = len(history_messages) // 2

        # Save conversation turn to Firestore (async in background)
        threading.Thread(
            target=chat_memory_service.save_conversation_turn,
            args=(user_id, session_id, user_message, ai_response, turn_number),
        ).start()

        # Update session activity (async in background)
        threading.Thread(
            target=chat_memory_service.update_session_activity,
            args=(user_id, session_id),
        ).start()

        # Update message count in cache
        if (
            user_id in active_chat_sessions
            and session_id in active_chat_sessions[user_id]
        ):
            active_chat_sessions[user_id][session_id]["message_count"] += 1
            active_chat_sessions[user_id][session_id][
                "last_activity"
            ] = datetime.utcnow()

        # Return response
        timestamp = datetime.utcnow().isoformat()
        return (
            jsonify(
                {
                    "success": True,
                    "user_message": {
                        "content": user_message,
                        "timestamp": timestamp,
                        "role": "user",
                    },
                    "ai_response": {
                        "content": ai_response,
                        "timestamp": timestamp,
                        "role": "assistant",
                    },
                    "session_id": session_id,
                    "metadata": {
                        "message_count": active_chat_sessions.get(user_id, {})
                        .get(session_id, {})
                        .get("message_count", 0),
                    },
                }
            ),
            201,
        )

    except Exception as e:
        print(f"Error in send_chat_message: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Failed to process message",
                    "message": str(e),
                }
            ),
            500,
        )


# ============================================================================
# SESSION MANAGEMENT ENDPOINT
# ============================================================================


@chat_bp.route("/end-session", methods=["POST"])
@token_required
def end_chat_session(user_id):
    """
    End a chat session and cleanup server-side resources.

    Expected JSON payload:
    {
        "session_id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6"
    }

    Returns:
        JSON response confirming session cleanup
    """
    try:
        # Get and validate JSON data
        request_data = request.get_json()
        if not request_data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid request",
                        "message": "JSON data is required",
                    }
                ),
                400,
            )

        # Extract and validate session_id
        session_id = request_data.get("session_id", "").strip()
        if not session_id:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Missing session_id",
                        "message": "session_id is required",
                    }
                ),
                400,
            )

        cleanup_count = 0

        # Clean up in-memory session
        if (
            user_id in active_chat_sessions
            and session_id in active_chat_sessions[user_id]
        ):
            del active_chat_sessions[user_id][session_id]
            cleanup_count += 1

        # Mark session as ended in Firestore (async)
        threading.Thread(
            target=chat_memory_service.end_session, args=(user_id, session_id)
        ).start()

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Session ended successfully",
                    "cleanup_count": cleanup_count,
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "error": "Failed to end session", "message": str(e)}
            ),
            500,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _load_session(session_id, user_id):
    """
    Load session data including history and LCEL chain.

    Args:
        session_id (str): Session ID or None
        user_id (str): User ID from authentication

    Returns:
        tuple: (session_id, chain, history_messages)
    """
    try:
        # Initialize user's session dictionary if it doesn't exist
        if user_id not in active_chat_sessions:
            active_chat_sessions[user_id] = {}

        # Case 1: No session_id provided - create new session
        if not session_id:
            session_id = str(uuid.uuid4())

            # Create session metadata in Firestore (async)
            threading.Thread(
                target=chat_memory_service.create_session_metadata,
                args=(user_id, session_id),
            ).start()

            # Create fresh chain
            chain = _create_chain_for_user(user_id)

            # Store in cache
            active_chat_sessions[user_id][session_id] = {
                "chain": chain,
                "last_activity": datetime.utcnow(),
                "message_count": 0,
            }

            return session_id, chain, []

        # Case 2: Session_id provided

        # Load history from Firestore (always load fresh history to be safe)
        # We convert the raw dicts to LangChain Message objects
        raw_messages = chat_memory_service.load_conversation_memory(
            user_id, session_id, max_messages=25
        )
        history_messages = []

        for msg in raw_messages:
            # Assuming msg has 'role' and 'content' or similar structure from load_conversation_memory
            # We need to adapt based on what chat_memory_service returns
            # If it returns LangChain messages, great. If dicts, we convert.
            if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
                history_messages.append(msg)
            elif isinstance(msg, dict):
                # Adapt based on your Firestore structure
                role = msg.get("role") or msg.get("type")
                content = msg.get("content") or msg.get("text")
                if role == "user":
                    history_messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    history_messages.append(AIMessage(content=content))

        # Check cache for existing chain
        if session_id in active_chat_sessions[user_id]:
            chain = active_chat_sessions[user_id][session_id]["chain"]
            return session_id, chain, history_messages

        # If not in cache, create new chain
        chain = _create_chain_for_user(user_id)

        # Store in cache
        active_chat_sessions[user_id][session_id] = {
            "chain": chain,
            "last_activity": datetime.utcnow(),
            "message_count": len(history_messages) // 2,
        }

        return session_id, chain, history_messages

    except Exception as e:
        print(f"Error in _load_session: {str(e)}")
        return None, None, []


def _create_chain_for_user(user_id):
    """
    Helper to create a personalized LCEL chain for a user.
    """
    # Fetch user context
    user_profile = chat_memory_service.get_user_profile(user_id)
    mood_logs = mood_service.get_last_daily_mood_logs(user_id, limit=7)

    # Generate mood summary if logs exist
    mood_summary = ""
    if mood_logs:
        mood_summary = llm_service.summarize_mood_logs(mood_logs)

    # Build user context
    user_context = llm_service.build_user_context(user_profile, mood_summary)

    # Create chain
    return llm_service.create_conversation_chain(user_context)
