"""Daily mood quiz endpoint only."""

from flask import Blueprint, request, jsonify
from stressease.services.utility.auth_service import token_required
from stressease.services.mood.mood_service import (
    save_daily_mood_log,
    get_last_daily_mood_logs,
    get_daily_mood_logs_count,
    weekly_dass_exists,
    save_weekly_dass_totals,
)
from datetime import datetime, date

# Create the mood blueprint
mood_bp = Blueprint("mood", __name__)


# ******************************************************************************
# * POST /api/mood/quiz/daily - Submit structured daily mood quiz
# ******************************************************************************
@mood_bp.route("/quiz/daily", methods=["POST"])
@token_required
def submit_daily_quiz(user_id):
    """
    Submit the 12-question daily mood quiz, compute stats, and store to Firestore.

    Expected JSON payload:
    {
      "core_scores": {"mood":1-5, "energy":1-5, "sleep":1-5, "stress":1-5},
      "rotating_scores": {"domain_name":"social", "scores":[1-5,1-5,1-5,1-5,1-5]},
      "dass_today": {"depression":1-5, "anxiety":1-5, "stress":1-5},
      "date": "YYYY-MM-DD" (optional),
      "additional_notes": "..." (optional)
    }

    Returns:
      { success, log_id, high_point, low_point, maybe weekly_dass if computed }
    """
    try:
        payload = request.get_json()
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

        # Validate required sections
        core = payload.get("core_scores")
        rotating = payload.get("rotating_scores")
        dass = payload.get("dass_today")

        if not core or not rotating or not dass:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Missing required fields",
                        "message": "core_scores, rotating_scores, and dass_today are required",
                    }
                ),
                400,
            )

        # Validate core scores
        required_core_keys = ["mood", "energy", "sleep", "stress"]
        if any(k not in core for k in required_core_keys):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid core_scores",
                        "message": "Missing one of mood, energy, sleep, stress",
                    }
                ),
                400,
            )
        core_scores = [core["mood"], core["energy"], core["sleep"], core["stress"]]
        if not all(isinstance(s, int) and 1 <= s <= 5 for s in core_scores):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid core_scores",
                        "message": "All core scores must be integers between 1 and 5",
                    }
                ),
                400,
            )

        # Validate rotating scores
        if "domain_name" not in rotating or "scores" not in rotating:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid rotating_scores",
                        "message": "domain_name and scores are required",
                    }
                ),
                400,
            )
        if not isinstance(rotating["scores"], list) or len(rotating["scores"]) != 5:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid rotating_scores",
                        "message": "scores must be a list of 5 integers",
                    }
                ),
                400,
            )
        rotating_scores = rotating["scores"]
        if not all(isinstance(s, int) and 1 <= s <= 5 for s in rotating_scores):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid rotating_scores",
                        "message": "All rotating scores must be integers between 1 and 5",
                    }
                ),
                400,
            )

        # Validate DASS today
        required_dass_keys = ["depression", "anxiety", "stress"]
        if any(k not in dass for k in required_dass_keys):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid dass_today",
                        "message": "Missing one of depression, anxiety, stress",
                    }
                ),
                400,
            )
        dass_scores = [dass["depression"], dass["anxiety"], dass["stress"]]
        if not all(isinstance(s, int) and 1 <= s <= 5 for s in dass_scores):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid dass_today",
                        "message": "All DASS scores must be integers between 1 and 5",
                    }
                ),
                400,
            )

        # Step 2 — Compute Daily Averages (1–5 scale)
        core_avg = sum(core_scores) / len(core_scores)
        rotating_avg = sum(rotating_scores) / len(rotating_scores)

        # Step 3 — Identify High & Low Points
        all_scores = (
            core_scores
            + rotating_scores
            + [dass["depression"], dass["anxiety"], dass["stress"]]
        )
        all_questions = [
            "q1",
            "q2",
            "q3",
            "q4",  # core
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",  # rotating
            "q10",
            "q11",
            "q12",  # DASS
        ]
        high_idx = all_scores.index(max(all_scores))
        low_idx = all_scores.index(min(all_scores))
        high_point = {
            "question_id": all_questions[high_idx],
            "score": all_scores[high_idx],
        }
        low_point = {
            "question_id": all_questions[low_idx],
            "score": all_scores[low_idx],
        }

        # Build Firestore document
        daily_doc = {
            "date": payload.get("date"),
            "core_scores": {
                "mood": core["mood"],
                "energy": core["energy"],
                "sleep": core["sleep"],
                "stress": core["stress"],
            },
            "rotating_scores": {
                "domain_name": rotating["domain_name"],
                "scores": rotating_scores,
            },
            "dass_today": {
                "depression": dass["depression"],
                "anxiety": dass["anxiety"],
                "stress": dass["stress"],
            },
            "high_point": high_point,
            "low_point": low_point,
            # Optionally include averages for analytics convenience
            "core_avg": core_avg,
            "rotating_avg": rotating_avg,
        }
        if "additional_notes" in payload and payload["additional_notes"]:
            daily_doc["additional_notes"] = payload["additional_notes"]

        # Save daily log
        log_id = save_daily_mood_log(user_id, daily_doc)
        if not log_id:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Database error",
                        "message": "Failed to save daily mood log",
                    }
                ),
                500,
            )

        # After saving, check if we have 7 logs to trigger weekly DASS aggregation
        weekly_result = None
        # Only trigger when total count is a multiple of 7 (i.e., end of a 7-day block)
        total_count = get_daily_mood_logs_count(user_id)
        last_7 = get_last_daily_mood_logs(user_id, 7)
        if total_count >= 7 and total_count % 7 == 0 and len(last_7) == 7:
            # Extract DASS series
            def _to_dass_scale(score: int) -> int:
                mapping = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}
                return mapping[int(score)]

            depression_vals = [
                entry.get("dass_today", {}).get("depression", 0) for entry in last_7
            ]
            anxiety_vals = [
                entry.get("dass_today", {}).get("anxiety", 0) for entry in last_7
            ]
            stress_vals = [
                entry.get("dass_today", {}).get("stress", 0) for entry in last_7
            ]

            depression_total = sum(_to_dass_scale(s) for s in depression_vals) * 2
            anxiety_total = sum(_to_dass_scale(s) for s in anxiety_vals) * 2
            stress_total = sum(_to_dass_scale(s) for s in stress_vals) * 2

            # Determine week range based on earliest/latest dates in the 7 logs
            # Prefer explicit 'date' field, else derive from submitted_at
            def _extract_date(entry):
                d = entry.get("date")
                if d:
                    return d
                ts = entry.get("submitted_at")
                try:
                    # Firestore returns datetime for server timestamps
                    if isinstance(ts, datetime):
                        return ts.date().isoformat()
                except Exception:
                    pass
                return None

            dates = [d for d in (_extract_date(e) for e in last_7) if d]
            if dates:
                week_start = min(dates)
                week_end = max(dates)
            else:
                # Fallback to today if dates missing
                week_start = date.today().isoformat()
                week_end = week_start

            # Avoid duplicates: only save if a record for this range doesn't exist
            if not weekly_dass_exists(user_id, week_start, week_end):
                weekly_id = save_weekly_dass_totals(
                    user_id,
                    week_start,
                    week_end,
                    depression_total,
                    anxiety_total,
                    stress_total,
                )
                if weekly_id:
                    # Compute optional weekly summary for response (not stored)
                    core_avgs_per_day = []
                    rotating_avgs_per_day = []
                    for entry in last_7:
                        c = entry.get("core_scores", {})
                        r = entry.get("rotating_scores", {})
                        cs = [
                            c.get("mood", 0),
                            c.get("energy", 0),
                            c.get("sleep", 0),
                            c.get("stress", 0),
                        ]
                        rs = r.get("scores", [])
                        if len(cs) == 4:
                            core_avgs_per_day.append(sum(cs) / 4)
                        if isinstance(rs, list) and len(rs) == 5:
                            rotating_avgs_per_day.append(sum(rs) / 5)

                    weekly_core_avg = (
                        round(sum(core_avgs_per_day) / len(core_avgs_per_day), 2)
                        if core_avgs_per_day
                        else None
                    )
                    weekly_rotating_avg = (
                        round(
                            sum(rotating_avgs_per_day) / len(rotating_avgs_per_day), 2
                        )
                        if rotating_avgs_per_day
                        else None
                    )

                    weekly_result = {
                        "weekly_id": weekly_id,
                        "week_start": week_start,
                        "week_end": week_end,
                        "depression_total": depression_total,
                        "anxiety_total": anxiety_total,
                        "stress_total": stress_total,
                        "weekly_core_avg": weekly_core_avg,
                        "weekly_rotating_avg": weekly_rotating_avg,
                    }

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Daily mood quiz saved successfully",
                    "log_id": log_id,
                    "high_point": high_point,
                    "low_point": low_point,
                    "weekly_dass": weekly_result,
                }
            ),
            201,
        )

    except Exception as e:
        return (
            jsonify({"success": False, "error": "Server error", "message": str(e)}),
            500,
        )
