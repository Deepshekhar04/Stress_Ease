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


def save_daily_mood_log(
    user_id: str, daily_log: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Save a structured daily mood quiz log to Firestore using composite document ID.

    Uses {user_id}_{date} as document ID to ensure one entry per user per day.
    Updates existing document if user submits multiple times in same day.

    Collection: user_mood_logs

    Args:
        user_id (str): Firebase Auth user ID
        daily_log (dict): Structured daily log payload

    Returns:
        Optional[Dict]: {
            "doc_id": str,
            "is_update": bool,
            "submission_count": int
        } if saved successfully, else None
    """
    db = get_firestore_client()

    try:
        # Ensure date is set (use client-provided date for timezone accuracy)
        if "date" not in daily_log or not daily_log["date"]:
            daily_log["date"] = date.today().isoformat()

        # Create composite document ID: {user_id}_{date}
        doc_id = f"{user_id}_{daily_log['date']}"
        doc_ref = db.collection("user_mood_logs").document(doc_id)

        # Check if document already exists
        existing = doc_ref.get()
        is_update = existing.exists

        # Set user_id
        daily_log["user_id"] = user_id

        # Handle timestamps and submission tracking
        if is_update:
            # Preserve first submission time, update last modified time
            existing_data = existing.to_dict()
            daily_log["first_submitted_at"] = existing_data.get(
                "first_submitted_at", datetime.utcnow()
            )
            daily_log["last_updated_at"] = datetime.utcnow()
            daily_log["submission_count"] = existing_data.get("submission_count", 1) + 1
        else:
            # First submission
            daily_log["first_submitted_at"] = datetime.utcnow()
            daily_log["last_updated_at"] = datetime.utcnow()
            daily_log["submission_count"] = 1

        # Set document (full replace)
        doc_ref.set(daily_log)

        return {
            "doc_id": doc_id,
            "is_update": is_update,
            "submission_count": daily_log["submission_count"],
        }
    except Exception as e:
        print(f"Error saving daily mood log for {user_id}: {str(e)}")
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


# ============================================================================
# DAILY QUIZ QUESTION FETCHING
# ============================================================================


def get_daily_questions(day_key: str) -> List[Dict[str, Any]]:
    """
    Fetch daily quiz questions from Firestore for context-aware AI insights.

    Args:
        day_key (str): Day identifier (e.g., "day_1", "day_2")

    Returns:
        List[Dict]: List of question objects with 'text', 'dimension', 'options', etc.
                    Returns empty list if not found or invalid structure.
    """
    db = get_firestore_client()

    try:
        doc_ref = db.collection("questions").document(day_key)
        doc = doc_ref.get()

        if not doc.exists:
            print(f"⚠ No questions document found for {day_key}")
            return []

        data = doc.to_dict()
        questions_field = data.get("questions", [])

        # Handle both List and Map structures (matching frontend logic)
        if isinstance(questions_field, list):
            # Direct list of questions
            return questions_field
        elif isinstance(questions_field, dict):
            # Map structure - extract values and sort by key if numeric
            try:
                # Try to sort numerically if keys are numbers
                sorted_keys = sorted(
                    questions_field.keys(), key=lambda x: int(x) if x.isdigit() else x
                )
                return [questions_field[k] for k in sorted_keys]
            except (ValueError, TypeError):
                # Fallback to values without sorting
                return list(questions_field.values())
        else:
            print(
                f"⚠ Unexpected questions field structure for {day_key}: {type(questions_field)}"
            )
            return []

    except Exception as e:
        print(f"✗ Error fetching questions for {day_key}: {str(e)}")
        return []


# ============================================================================
# DAILY QUIZ DUPLICATE PREVENTION
# ============================================================================


def get_daily_mood_log_by_date(user_id: str, date_str: str) -> Optional[Dict[str, Any]]:
    """
    Check if a mood log already exists for a specific date.

    Args:
        user_id (str): Firebase Auth user ID
        date_str (str): ISO date string (YYYY-MM-DD)

    Returns:
        Optional[Dict]: Existing log with 'id' field, or None if not found
    """
    db = get_firestore_client()

    try:
        query = (
            db.collection("user_mood_logs")
            .where("user_id", "==", user_id)
            .where("date", "==", date_str)
            .limit(1)
        )
        docs = query.stream()
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            return data
        return None
    except Exception as e:
        print(f"Error checking daily mood log for {user_id} on {date_str}: {str(e)}")
        return None


def update_daily_mood_log(doc_id: str, daily_log: Dict[str, Any]) -> bool:
    """
    Update an existing daily mood log.

    Args:
        doc_id (str): Document ID to update
        daily_log (dict): Updated log data

    Returns:
        bool: True if successful, False otherwise
    """
    db = get_firestore_client()

    try:
        daily_log["submitted_at"] = datetime.utcnow()  # Update timestamp
        db.collection("user_mood_logs").document(doc_id).update(daily_log)
        return True
    except Exception as e:
        print(f"Error updating daily mood log {doc_id}: {str(e)}")
        return False
