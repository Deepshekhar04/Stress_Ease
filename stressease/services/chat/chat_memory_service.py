"""
Chat session and conversation memory service.

This module handles all chat session operations including:
- Session metadata management
- Conversation memory persistence
- Message history storage
- User profile retrieval
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from stressease.services.utility.firebase_config import get_firestore_client


# ============================================================================
# USER PROFILE OPERATIONS
# ============================================================================


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user profile from Firestore.

    Collection: users/{user_id}/profile or users/{user_id}

    Args:
        user_id (str): Firebase Auth user ID

    Returns:
        Optional[Dict]: User profile data or None if not found
    """
    db = get_firestore_client()

    try:
        # Try structure: users/{user_id}/profile/data
        doc_ref = (
            db.collection("users")
            .document(user_id)
            .collection("profile")
            .document("data")
        )
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()

        # Try alternative structure: users/{user_id} (profile as document)
        user_doc = db.collection("users").document(user_id).get()
        if user_doc.exists:
            return user_doc.to_dict()

        return None

    except Exception as e:
        print(f"Error fetching user profile for {user_id}: {str(e)}")
        return None


# ============================================================================
# CHAT SESSION OPERATIONS
# ============================================================================


def create_session_metadata(user_id: str, session_id: str) -> bool:
    """
    Create initial session metadata document in Firestore.

    Collection: users/{user_id}/chat_sessions/{session_id}/metadata/info

    Args:
        user_id (str): Firebase Auth user ID
        session_id (str): Unique session identifier

    Returns:
        bool: True if successful, False otherwise
    """
    db = get_firestore_client()

    try:
        metadata = {
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "status": "active",
            "message_count": 0,
        }

        db.collection("users").document(user_id).collection("chat_sessions").document(
            session_id
        ).collection("metadata").document("info").set(metadata)

        return True

    except Exception as e:
        print(f"Error creating session metadata for {user_id}/{session_id}: {str(e)}")
        return False


def update_session_activity(user_id: str, session_id: str) -> bool:
    """
    Update last_activity timestamp and increment message count for a session.

    Collection: users/{user_id}/chat_sessions/{session_id}/metadata/info

    Args:
        user_id (str): Firebase Auth user ID
        session_id (str): Session identifier

    Returns:
        bool: True if successful, False otherwise
    """
    db = get_firestore_client()

    try:
        from firebase_admin import firestore

        metadata_ref = (
            db.collection("users")
            .document(user_id)
            .collection("chat_sessions")
            .document(session_id)
            .collection("metadata")
            .document("info")
        )

        # Update last activity and increment message count
        metadata_ref.update(
            {
                "last_activity": datetime.utcnow(),
                "message_count": firestore.Increment(1),
            }
        )

        return True

    except Exception as e:
        print(f"Error updating session activity for {user_id}/{session_id}: {str(e)}")
        return False


def check_session_expired(
    user_id: str, session_id: str, expiry_hours: int = 24
) -> bool:
    """
    Check if a session has expired based on last_activity timestamp.

    Args:
        user_id (str): Firebase Auth user ID
        session_id (str): Session identifier
        expiry_hours (int): Hours of inactivity before expiry (default: 24)

    Returns:
        bool: True if expired, False if still active
    """
    db = get_firestore_client()

    try:
        metadata_ref = (
            db.collection("users")
            .document(user_id)
            .collection("chat_sessions")
            .document(session_id)
            .collection("metadata")
            .document("info")
        )

        doc = metadata_ref.get()

        if not doc.exists:
            return True  # No metadata = expired/invalid session

        data = doc.to_dict()
        last_activity = data.get("last_activity")

        if not last_activity:
            return True  # No activity timestamp = expired

        # Check if last activity was more than expiry_hours ago
        expiry_threshold = datetime.utcnow() - timedelta(hours=expiry_hours)

        return last_activity < expiry_threshold

    except Exception as e:
        print(f"Error checking session expiry for {user_id}/{session_id}: {str(e)}")
        return True  # Assume expired on error


def end_session(user_id: str, session_id: str) -> bool:
    """
    Mark a session as ended and update its status.

    Args:
        user_id (str): Firebase Auth user ID
        session_id (str): Session identifier

    Returns:
        bool: True if successful, False otherwise
    """
    db = get_firestore_client()

    try:
        metadata_ref = (
            db.collection("users")
            .document(user_id)
            .collection("chat_sessions")
            .document(session_id)
            .collection("metadata")
            .document("info")
        )

        metadata_ref.update({"status": "ended", "ended_at": datetime.utcnow()})

        return True

    except Exception as e:
        print(f"Error ending session for {user_id}/{session_id}: {str(e)}")
        return False


# ============================================================================
# CONVERSATION MEMORY OPERATIONS
# ============================================================================


def load_conversation_memory(
    user_id: str, session_id: str, max_messages: int = 25
) -> List[BaseMessage]:
    """
    Load conversation history from Firestore and convert to LangChain message format.

    Collection: users/{user_id}/chat_sessions/{session_id}/messages

    Args:
        user_id (str): Firebase Auth user ID
        session_id (str): Session identifier
        max_messages (int): Maximum number of messages to load (default: 25)

    Returns:
        List[BaseMessage]: List of LangChain HumanMessage and AIMessage objects
    """
    db = get_firestore_client()

    try:
        from firebase_admin import firestore

        # Query messages ordered by timestamp, get last max_messages
        query = (
            db.collection("users")
            .document(user_id)
            .collection("chat_sessions")
            .document(session_id)
            .collection("messages")
            .order_by("timestamp", direction=firestore.Query.ASCENDING)
            .limit(max_messages)
        )

        docs = query.stream()
        messages = []

        for doc in docs:
            data = doc.to_dict()
            role = data.get("role", "")
            content = data.get("content", "")

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

    except Exception as e:
        print(f"Error loading conversation memory for {user_id}/{session_id}: {str(e)}")
        return []


def save_conversation_turn(
    user_id: str, session_id: str, user_msg: str, ai_msg: str, turn_number: int
) -> bool:
    """
    Save one conversation exchange (user message + AI response) to Firestore.

    Collection: users/{user_id}/chat_sessions/{session_id}/messages

    Args:
        user_id (str): Firebase Auth user ID
        session_id (str): Session identifier
        user_msg (str): User's message content
        ai_msg (str): AI's response content
        turn_number (int): Turn number in conversation

    Returns:
        bool: True if successful, False otherwise
    """
    db = get_firestore_client()

    try:
        timestamp = datetime.utcnow()
        messages_ref = (
            db.collection("users")
            .document(user_id)
            .collection("chat_sessions")
            .document(session_id)
            .collection("messages")
        )

        # Save user message
        user_message_data = {
            "role": "user",
            "content": user_msg,
            "timestamp": timestamp,
            "turn": turn_number,
        }
        messages_ref.add(user_message_data)

        # Save AI message
        ai_message_data = {
            "role": "assistant",
            "content": ai_msg,
            "timestamp": timestamp,
            "turn": turn_number,
        }
        messages_ref.add(ai_message_data)

        return True

    except Exception as e:
        print(f"Error saving conversation turn for {user_id}/{session_id}: {str(e)}")
        return False
