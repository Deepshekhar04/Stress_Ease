"""Stress prediction endpoint."""

from flask import Blueprint, request, jsonify
from typing import Optional, Tuple
from stressease.services.utility.auth_service import token_required
from stressease.services.prediction.prediction_service import predict_stress

# Create the predict blueprint
predict_bp = Blueprint("predict", __name__)


def _calculate_avg_quiz_score(user_id: str) -> Tuple[Optional[float], int]:
    """
    Calculate 7-day average of daily_total_score from Firestore.

    Args:
        user_id: Firebase Auth user ID

    Returns:
        Tuple of (avg_score, days_count):
            - avg_score: Average quiz score (0-60), or None if no data
            - days_count: Number of days of data available
    """
    from stressease.services.mood.mood_service import get_last_daily_mood_logs

    logs = get_last_daily_mood_logs(user_id, limit=7)

    if not logs:
        return None, 0

    # Extract daily_total_score from each log
    scores = []
    for log in logs:
        score = log.get("daily_total_score")
        if score is not None:
            scores.append(score)

    if not scores:
        return None, 0

    avg = sum(scores) / len(scores)
    days_count = len(scores)

    print(f"✓ Calculated avgQuizScore from {days_count} day(s): {avg:.2f}")
    return round(avg, 2), days_count


# ******************************************************************************
# * POST /api/predict - Predict tomorrow's stress level
# ******************************************************************************
@predict_bp.route("/predict", methods=["POST"])
@token_required
def predict(user_id):
    """
    Predict tomorrow's stress level based on 7-day metrics.

    Expected JSON payload:
    {
        "avgMoodScore": 2.3,      // Float: 1.0 - 5.0 (7-day average)
        "chatCount": 8,            // Integer: 0 - 999 (chat sessions in last 7 days)
        "avgQuizScore": 24         // Integer: 0 - 60 (avg sum of 12 questions over 7 days)
    }

    Returns:
    {
        "success": true,
        "prediction": {
            "date": "2025-12-13",
            "stressProbability": 0.76,
            "label": "High",
            "confidence": 0.73,
            "basedOn": {
                "avgMoodScore": 2.3,
                "chatCount": 8,
                "avgQuizScore": 24
            }
        }
    }
    """
    try:
        # DEBUG: Log incoming request details
        print(f"\n{'='*60}")
        print(f"📥 Incoming /api/predict request from user: {user_id}")
        print(f"Headers: {dict(request.headers)}")
        print(f"{'='*60}\n")

        payload = request.get_json()
        print(f"📦 Payload received: {payload}")

        if not payload:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid request",
                        "message": "JSON body required",
                    }
                ),
                400,
            )

        # Extract required fields from frontend
        avg_mood_score = payload.get("avgMoodScore")
        chat_count = payload.get("chatCount")
        frontend_avg_quiz_score = payload.get(
            "avgQuizScore"
        )  # Optional, backend will calculate

        # Calculate avgQuizScore from backend (source of truth)
        backend_avg_quiz_score, quiz_data_days = _calculate_avg_quiz_score(user_id)

        # Determine which score to use with graceful fallback
        if backend_avg_quiz_score is not None:
            # Backend calculation succeeded
            avg_quiz_score = backend_avg_quiz_score
            data_source = "backend"

            # Compare with frontend value if provided (for debugging)
            if frontend_avg_quiz_score is not None:
                diff = abs(frontend_avg_quiz_score - backend_avg_quiz_score)
                if diff > 0.5:  # Allow small rounding differences
                    print(f"⚠️ Quiz score mismatch for user {user_id}:")
                    print(f"   Frontend: {frontend_avg_quiz_score}")
                    print(f"   Backend: {backend_avg_quiz_score} (using this)")
                    print(f"   Difference: {diff:.2f}")
        else:
            # Backend calculation failed - fallback to frontend
            if frontend_avg_quiz_score is None:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Insufficient Data",
                            "message": "Please complete at least one daily quiz before requesting predictions",
                        }
                    ),
                    400,
                )

            print(
                f"⚠️ Backend quiz calculation failed, using frontend value: {frontend_avg_quiz_score}"
            )
            avg_quiz_score = frontend_avg_quiz_score
            data_source = "frontend"
            quiz_data_days = 0  # Unknown

        # Validate other required fields (avgQuizScore already calculated above)
        if avg_mood_score is None or chat_count is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Missing required fields",
                        "message": "avgMoodScore and chatCount are required",
                    }
                ),
                400,
            )

        # Validate avgMoodScore
        try:
            avg_mood_score = float(avg_mood_score)
            if avg_mood_score < 1.0 or avg_mood_score > 5.0:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Invalid Input",
                            "message": "avgMoodScore must be between 1.0 and 5.0",
                        }
                    ),
                    400,
                )
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid Input",
                        "message": "avgMoodScore must be a number between 1.0 and 5.0",
                    }
                ),
                400,
            )

        # Validate chatCount
        try:
            chat_count = int(chat_count)
            if chat_count < 0 or chat_count > 999:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Invalid Input",
                            "message": "chatCount must be between 0 and 999",
                        }
                    ),
                    400,
                )
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid Input",
                        "message": "chatCount must be an integer between 0 and 999",
                    }
                ),
                400,
            )


        # Call prediction service
        prediction = predict_stress(avg_mood_score, chat_count, avg_quiz_score)

        # Add data quality metadata to prediction
        prediction["dataQuality"] = {
            "quizDataDays": quiz_data_days,
            "quizDataSource": data_source,
        }

        # Return successful response
        return (
            jsonify(
                {
                    "success": True,
                    "prediction": prediction,
                }
            ),
            200,
        )

    except Exception as e:
        print(f"✗ Error in /api/predict for user {user_id}: {str(e)}")
        return (
            jsonify({"success": False, "error": "Server error", "message": str(e)}),
            500,
        )
