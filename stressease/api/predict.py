"""Stress prediction endpoint."""

from flask import Blueprint, request, jsonify
from stressease.services.utility.auth_service import token_required
from stressease.services.prediction.prediction_service import predict_stress

# Create the predict blueprint
predict_bp = Blueprint("predict", __name__)


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

        # Extract required fields
        avg_mood_score = payload.get("avgMoodScore")
        chat_count = payload.get("chatCount")
        avg_quiz_score = payload.get("avgQuizScore")

        # Validate all required fields are present
        if avg_mood_score is None or chat_count is None or avg_quiz_score is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Missing required fields",
                        "message": "avgMoodScore, chatCount, and avgQuizScore are required",
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

        # Validate avgQuizScore
        try:
            avg_quiz_score = int(avg_quiz_score)
            if avg_quiz_score < 0 or avg_quiz_score > 60:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Invalid Input",
                            "message": "avgQuizScore must be between 0 and 60",
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
                        "message": "avgQuizScore must be an integer between 0 and 60",
                    }
                ),
                400,
            )

        # Call prediction service
        prediction = predict_stress(avg_mood_score, chat_count, avg_quiz_score)

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
