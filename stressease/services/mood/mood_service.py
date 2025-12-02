"""
Mood quiz and DASS operations service.

This module handles all mood tracking operations including:
- Daily mood quiz logs
- Weekly DASS aggregation
- Mood history retrieval
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any
from stressease.services.utility.firebase_config import get_firestore_client


# ============================================================================
# MOOD QUIZ OPERATIONS
# ============================================================================


def upsert_daily_mood_log(
    user_id: str, daily_log: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Save or update daily mood quiz using Firestore's upsert pattern.

    Uses composite document ID: {user_id}_{date}
    This ensures one quiz per user per day without explicit checking.

    Single operation replaces the old check-then-save/update pattern,
    improving performance by ~100ms and reducing Firestore operations by 50%.

    Args:
        user_id (str): Firebase Auth user ID
        daily_log (dict): Daily mood quiz data

    Returns:
        Optional[Dict]: {
            "doc_id": str,
            "date": str,
            "operation": "upsert"
        } if saved successfully, else None
    """
    db = get_firestore_client()

    try:
        # Ensure date is set (use provided or today)
        date_str = daily_log.get("date") or date.today().isoformat()
        daily_log["date"] = date_str

        # Composite document ID: {user_id}_{date}
        doc_id = f"{user_id}_{date_str}"

        # Set user_id
        daily_log["user_id"] = user_id

        # Add server timestamp
        daily_log["submitted_at"] = datetime.utcnow()

        # Upsert: creates if new, overwrites if exists
        # This is naturally idempotent - same request multiple times = same result
        doc_ref = db.collection("user_mood_logs").document(doc_id)
        doc_ref.set(daily_log)

        print(f"✓ Upserted mood log: {doc_id}")

        return {"doc_id": doc_id, "date": date_str, "operation": "upsert"}

    except Exception as e:
        print(f"✗ Error upserting mood log for {user_id}: {str(e)}")
        return None


def get_last_daily_mood_logs(user_id: str, limit: int = 7) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent daily mood quiz logs for a user.

    Args:
        user_id (str): Firebase Auth user ID
        limit (int): Number of entries to retrieve (default: 7)

    Returns:
        List[Dict[str, Any]]: List of daily mood logs (newest first)
    """
    db = get_firestore_client()

    try:
        from firebase_admin import firestore

        logs = []
        query = (
            db.collection("user_mood_logs")
            .where("user_id", "==", user_id)
            .order_by("submitted_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = query.stream()
        for doc in docs:
            entry = doc.to_dict()
            entry["id"] = doc.id
            logs.append(entry)
        return logs
    except Exception as e:
        print(f"Error retrieving last daily mood logs for {user_id}: {str(e)}")
        return []


def get_daily_mood_logs_count(user_id: str) -> int:
    """
    Count total number of daily mood logs for a user.

    Args:
        user_id (str): Firebase Auth user ID

    Returns:
        int: Total count of documents in user_mood_logs for the user
    """
    db = get_firestore_client()

    try:
        query = db.collection("user_mood_logs").where("user_id", "==", user_id)
        count = 0
        for _ in query.stream():
            count += 1
        return count
    except Exception as e:
        print(f"Error counting daily mood logs for {user_id}: {str(e)}")
        return 0


# ============================================================================
# WEEKLY DASS OPERATIONS
# ============================================================================


def weekly_dass_exists(user_id: str, week_start: str, week_end: str) -> bool:
    """
    Check if a weekly DASS record already exists for the given user and week.

    Args:
        user_id (str): Firebase Auth user ID
        week_start (str): ISO date string for week start
        week_end (str): ISO date string for week end

    Returns:
        bool: True if a record exists, False otherwise
    """
    db = get_firestore_client()

    try:
        query = (
            db.collection("user_weekly_dass")
            .where("user_id", "==", user_id)
            .where("week_start", "==", week_start)
            .where("week_end", "==", week_end)
        )
        docs = query.stream()
        for _ in docs:
            return True
        return False
    except Exception as e:
        print(f"Error checking weekly DASS existence for {user_id}: {str(e)}")
        return False


def save_weekly_dass_totals(
    user_id: str,
    week_start: str,
    week_end: str,
    depression_total: int,
    anxiety_total: int,
    stress_total: int,
) -> Optional[str]:
    """
    Save weekly DASS-21 totals to Firestore.

    Collection: user_weekly_dass

    Args:
        user_id (str): Firebase Auth user ID
        week_start (str): ISO date string for week start
        week_end (str): ISO date string for week end
        depression_total (int): Scaled total (DASS-21 x2)
        anxiety_total (int): Scaled total (DASS-21 x2)
        stress_total (int): Scaled total (DASS-21 x2)

    Returns:
        Optional[str]: Document ID if saved successfully, else None
    """
    db = get_firestore_client()

    try:
        data = {
            "user_id": user_id,
            "week_start": week_start,
            "week_end": week_end,
            "depression_total": depression_total,
            "anxiety_total": anxiety_total,
            "stress_total": stress_total,
            "calculated_at": datetime.utcnow(),
        }

        doc_ref = db.collection("user_weekly_dass").add(data)
        return doc_ref[1].id
    except Exception as e:
        print(f"Error saving weekly DASS totals for {user_id}: {str(e)}")
        return None
