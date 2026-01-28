"""Analytics API endpoints."""

from flask import Blueprint, request, jsonify
from stressease.services.utility.auth_service import token_required
from stressease.services.prediction.analytics_service import (
    fetch_analytics_data,
    calculate_summary,
    analyze_trends,
    generate_prediction,
)
import logging

logger = logging.getLogger(__name__)

# Create the analytics blueprint
analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/final-summary", methods=["POST"])
@token_required
def final_summary(user_id):
    """
    Generate final analytics summary from Firestore data.

    Backend fetches all data internally using the authenticated user_id.
    Frontend sends empty JSON body: {}

    Returns:
        JSON response with summary, trends, and prediction
    """
    try:
        # Step 1: Fetch analytics data from Firestore
        data_result = fetch_analytics_data(user_id, days=7)

        # Edge Case: No data available
        if not data_result["has_data"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "INSUFFICIENT_DATA",
                        "error": "Insufficient Data",
                        "message": "Please complete at least one daily quiz before viewing analytics",
                        "metadata": {
                            "days_analyzed": 0,
                            "data_quality": data_result["data_quality"],
                        },
                    }
                ),
                400,
            )

        mood_logs = data_result["mood_logs"]
        days_available = data_result["days_available"]
        data_quality = data_result["data_quality"]

        # Step 2: Calculate summary statistics
        summary = calculate_summary(mood_logs)

        # Step 3: Analyze trends
        trends = analyze_trends(mood_logs)

        # Step 4: Generate prediction (rule-based + LLM explanation)
        prediction = generate_prediction(summary, trends, mood_logs)

        # Step 5: Build response
        response_data = {
            "summary": summary,
            "trends": trends,
            "prediction": prediction,
            "metadata": {"days_analyzed": days_available, "data_quality": data_quality},
        }

        # Add recommendation for partial data
        if data_quality == "partial":
            response_data["metadata"][
                "recommendation"
            ] = "Complete more daily quizzes for better insights"

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(
            f"Error in /analytics/final-summary: {str(e)}",
            extra={"user_id": user_id},
            exc_info=True,
        )
        return (
            jsonify(
                {
                    "success": False,
                    "error_code": "SERVER_ERROR",
                    "error": "Server error",
                    "message": str(e),
                }
            ),
            500,
        )
