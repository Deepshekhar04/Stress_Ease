"""Analytics service for final summary endpoint.

Provides data aggregation, trend analysis, and hybrid prediction (rule-based + LLM).
"""

from typing import Dict, List, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from stressease.services.chat.llm_service import get_base_model


def fetch_analytics_data(user_id: str, days: int = 7) -> Dict:
    """
    Fetch mood logs with graceful degradation for incomplete data.

    Args:
        user_id: Firebase Auth user ID
        days: Number of days to fetch (default: 7)

    Returns:
        Dict containing:
            - has_data (bool): Whether any data exists
            - days_available (int): Number of days fetched
            - mood_logs (List[Dict]): The mood log documents
            - data_quality (str): 'complete', 'partial', or 'no_data'
    """
    from stressease.services.mood.mood_service import get_last_daily_mood_logs

    try:
        # Fetch last N days of mood logs
        mood_logs = get_last_daily_mood_logs(user_id, limit=days)

        # Edge Case 1: No data at all
        if not mood_logs:
            return {
                "has_data": False,
                "days_available": 0,
                "mood_logs": [],
                "data_quality": "no_data",
            }

        # Edge Case 2: Less than 7 days
        if len(mood_logs) < 7:
            return {
                "has_data": True,
                "days_available": len(mood_logs),
                "mood_logs": mood_logs,
                "data_quality": "partial",
            }

        # Ideal case: Full 7 days
        return {
            "has_data": True,
            "days_available": len(mood_logs),
            "mood_logs": mood_logs,
            "data_quality": "complete",
        }

    except Exception as e:
        print(f"Error fetching analytics data for {user_id}: {str(e)}")
        return {
            "has_data": False,
            "days_available": 0,
            "mood_logs": [],
            "data_quality": "error",
        }


def calculate_summary(mood_logs: List[Dict]) -> Dict:
    """
    Calculate avg_mood, avg_stress, and dominant_issue.

    Leverages pre-computed averages where available.

    Args:
        mood_logs: List of mood log documents from Firestore

    Returns:
        Dict with avg_mood, avg_stress, dominant_issue (all as strings)
    """
    mood_scores = []

    # DASS scores for avg_stress and dominant issue
    depression_scores = []
    anxiety_scores = []
    dass_stress_scores = []

    for log in mood_logs:
        # Use pre-computed core_avg if available (includes mood, energy, sleep, stress)
        # Otherwise fall back to individual mood score
        if "core_avg" in log:
            # core_avg is already the average of all 4 core questions
            # For mood specifically, we can still use the individual score
            mood_scores.append(log.get("core_scores", {}).get("mood", log["core_avg"]))
        elif "core_scores" in log and "mood" in log["core_scores"]:
            mood_scores.append(log["core_scores"]["mood"])

        # DASS subscales (Questions 10-12)
        dass = log.get("dass_today", {})
        if "depression" in dass:
            depression_scores.append(dass["depression"])
        if "anxiety" in dass:
            anxiety_scores.append(dass["anxiety"])
        if "stress" in dass:
            dass_stress_scores.append(dass["stress"])

    # Calculate averages
    avg_mood = round(sum(mood_scores) / len(mood_scores), 1) if mood_scores else None
    avg_stress = (
        round(sum(dass_stress_scores) / len(dass_stress_scores), 1)
        if dass_stress_scores
        else None
    )

    # Determine dominant issue (highest average score = biggest problem)
    issue_avgs = {
        "depression": (
            sum(depression_scores) / len(depression_scores) if depression_scores else 0
        ),
        "anxiety": sum(anxiety_scores) / len(anxiety_scores) if anxiety_scores else 0,
        "stress": (
            sum(dass_stress_scores) / len(dass_stress_scores)
            if dass_stress_scores
            else 0
        ),
    }

    dominant_issue = (
        max(issue_avgs, key=issue_avgs.get) if any(issue_avgs.values()) else "unknown"
    )

    return {
        "avg_mood": str(avg_mood) if avg_mood else "--",
        "avg_stress": str(avg_stress) if avg_stress else "--",
        "dominant_issue": dominant_issue,
    }


def analyze_trends(mood_logs: List[Dict]) -> Dict:
    """
    Determine if mood/stress is increasing, declining, or stable.

    Splits data into first half vs second half and compares averages.

    Args:
        mood_logs: List of mood log documents from Firestore

    Returns:
        Dict with mood and stress trends ('increasing', 'declining', or 'stable')
    """
    # Need at least 2 days for trend
    if len(mood_logs) < 2:
        return {"mood": "stable", "stress": "stable"}

    # Sort by date (oldest first)
    try:
        sorted_logs = sorted(
            mood_logs, key=lambda x: x.get("date", x.get("submitted_at", ""))
        )
    except Exception:
        # If sorting fails, use logs as-is
        sorted_logs = mood_logs

    # Split into first half and second half
    midpoint = len(sorted_logs) // 2
    first_half = sorted_logs[:midpoint]
    second_half = sorted_logs[midpoint:]

    # Calculate mood trend
    first_mood_avg = (
        sum(log.get("core_scores", {}).get("mood", 3) for log in first_half)
        / len(first_half)
        if first_half
        else 3.0
    )

    second_mood_avg = (
        sum(log.get("core_scores", {}).get("mood", 3) for log in second_half)
        / len(second_half)
        if second_half
        else 3.0
    )

    mood_diff = second_mood_avg - first_mood_avg

    # Mood: higher is better (1=worst, 5=best)
    if mood_diff > 0.5:
        mood_trend = "increasing"  # Getting happier
    elif mood_diff < -0.5:
        mood_trend = "declining"  # Getting sadder
    else:
        mood_trend = "stable"

    # Calculate stress trend
    first_stress_avg = (
        sum(log.get("dass_today", {}).get("stress", 3) for log in first_half)
        / len(first_half)
        if first_half
        else 3.0
    )

    second_stress_avg = (
        sum(log.get("dass_today", {}).get("stress", 3) for log in second_half)
        / len(second_half)
        if second_half
        else 3.0
    )

    stress_diff = second_stress_avg - first_stress_avg

    # Stress: higher is worse (1=best, 5=worst)
    if stress_diff > 0.5:
        stress_trend = "increasing"  # Getting more stressed
    elif stress_diff < -0.5:
        stress_trend = "declining"  # Getting less stressed
    else:
        stress_trend = "stable"

    return {"mood": mood_trend, "stress": stress_trend}


def generate_prediction(summary: Dict, trends: Dict, mood_logs: List[Dict]) -> Dict:
    """
    Hybrid approach: Rule-based prediction + LLM-generated explanation.

    Rules provide deterministic, reliable predictions.
    LLM provides human-relatable, contextual explanations.

    Args:
        summary: Summary dict from calculate_summary()
        trends: Trends dict from analyze_trends()
        mood_logs: Original mood log documents

    Returns:
        Dict with state, confidence, and reason
    """
    # STEP 1: Rule-based prediction (deterministic)
    state, confidence = _calculate_prediction_with_rules(
        summary, trends, len(mood_logs)
    )

    # STEP 2: LLM generates human-relatable explanation
    reason = _generate_explanation_with_llm(
        state, confidence, summary, trends, mood_logs
    )

    return {"state": state, "confidence": confidence, "reason": reason}


def _calculate_prediction_with_rules(
    summary: Dict, trends: Dict, days_count: int
) -> Tuple[str, str]:
    """
    Deterministic rule-based prediction.

    Args:
        summary: Summary statistics
        trends: Trend analysis results
        days_count: Number of days of data

    Returns:
        Tuple of (state, confidence)
    """
    mood_trend = trends["mood"]
    stress_trend = trends["stress"]

    # Parse averages (handle "--" for missing data)
    try:
        avg_mood = float(summary["avg_mood"]) if summary["avg_mood"] != "--" else 3.0
        avg_stress = (
            float(summary["avg_stress"]) if summary["avg_stress"] != "--" else 3.0
        )
    except (ValueError, TypeError):
        avg_mood = 3.0
        avg_stress = 3.0

    # Rule 1: Both improving → improving_wellbeing
    if mood_trend == "increasing" and stress_trend == "declining":
        state = "improving_wellbeing"

    # Rule 2: Both worsening → high_stress or increasing_stress
    elif mood_trend == "declining" and stress_trend == "increasing":
        if avg_stress >= 4.0 or avg_mood <= 2.0:
            state = "high_stress"
        else:
            state = "increasing_stress"

    # Rule 3: Stress increasing (regardless of mood) → concern
    elif stress_trend == "increasing":
        if avg_stress >= 4.0:
            state = "high_stress"
        else:
            state = "mild_concern"

    # Rule 4: Mood declining significantly → concern
    elif mood_trend == "declining" and avg_mood <= 2.5:
        state = "mild_concern"

    # Rule 5: Mood improving → improving
    elif mood_trend == "increasing":
        state = "improving_wellbeing"

    # Rule 6: Default stable state
    else:
        if avg_stress >= 4.0:
            state = "mild_concern"
        elif avg_mood >= 3.5 and avg_stress <= 3.0:
            state = "stable_wellbeing"
        else:
            state = "stable_wellbeing"

    # Determine confidence based on data availability
    if days_count >= 7:
        confidence = "high"
    elif days_count >= 4:
        confidence = "medium"
    else:
        confidence = "low"

    return state, confidence


def _generate_explanation_with_llm(
    state: str, confidence: str, summary: Dict, trends: Dict, mood_logs: List[Dict]
) -> str:
    """
    Use BASE model to generate human-relatable explanation.

    Falls back to template-based explanation if LLM fails.

    Args:
        state: Predicted state from rules
        confidence: Confidence level from rules
        summary: Summary statistics
        trends: Trend analysis
        mood_logs: Original mood logs

    Returns:
        Human-readable explanation string
    """
    # Prepare context for LLM
    context_data = {
        "state": state.replace("_", " ").title(),
        "confidence": confidence.title(),
        "avg_mood": f"{summary['avg_mood']}/5",
        "avg_stress": f"{summary['avg_stress']}/5",
        "dominant_issue": summary["dominant_issue"],
        "mood_trend": trends["mood"],
        "stress_trend": trends["stress"],
        "days_analyzed": len(mood_logs),
        "recent_days_summary": "",
    }

    # Add brief daily summary if available
    if mood_logs:
        recent = mood_logs[-3:]  # Last 3 days
        recent_summary = "\n**Recent Days:**"
        for i, log in enumerate(reversed(recent), 1):
            core = log.get("core_scores", {})
            dass = log.get("dass_today", {})
            recent_summary += f"\n  {i} day(s) ago: Mood {core.get('mood', '?')}/5, Stress {dass.get('stress', '?')}/5"
        context_data["recent_days_summary"] = recent_summary

    # Create LangChain PromptTemplate
    prompt = PromptTemplate(
        input_variables=[
            "state",
            "confidence",
            "avg_mood",
            "avg_stress",
            "dominant_issue",
            "mood_trend",
            "stress_trend",
            "days_analyzed",
            "recent_days_summary",
        ],
        template="""You are helping explain mental health analytics to a user.

**Prediction (already determined):**
- State: {state}
- Confidence: {confidence}

**User's Data Summary:**
- Average Mood: {avg_mood}
- Average Stress: {avg_stress}
- Dominant Issue: {dominant_issue}
- Mood Trend: {mood_trend}
- Stress Trend: {stress_trend}
- Days Analyzed: {days_analyzed}
{recent_days_summary}

Write a supportive 2-3 sentence explanation that helps the user understand their mental health prediction.

Requirements:
- Explain WHY we predicted "{state}" based on their trends
- Mention specific patterns you notice (mood/stress trends)
- Use warm, encouraging, non-clinical language
- Include a gentle suggestion (use chatbot, breathing exercises, or keep up good work)
- Respond with ONLY the explanation text, no JSON or formatting

Example good explanations:
- "Your mood has been improving and stress levels are declining over the past week. This positive trend suggests you're moving toward better wellbeing. Keep up the practices that are working for you!"
- "Your stress levels have been rising while your mood is declining. This pattern suggests you may be experiencing increasing stress. Consider trying the breathing exercises or chatting with our support bot."
""",
    )

    try:
        model = get_base_model()  # Use BASE model as requested

        # Create LCEL chain
        chain = prompt | model | StrOutputParser()

        # Invoke chain
        explanation = chain.invoke(context_data)

        # Basic validation - should be 2-3 sentences
        if len(explanation) < 20:
            raise ValueError("Explanation too short")

        return explanation

    except Exception as e:
        print(f"LLM explanation failed: {e}")
        # Fallback to template-based explanation
        return _template_based_explanation(state, confidence, summary, trends)


def _template_based_explanation(
    state: str, confidence: str, summary: Dict, trends: Dict
) -> str:
    """
    Fallback explanation templates if LLM fails.

    Args:
        state: Predicted state
        confidence: Confidence level
        summary: Summary statistics
        trends: Trend analysis

    Returns:
        Pre-defined template explanation
    """
    # Pre-defined explanation templates for each state
    templates = {
        "improving_wellbeing": "Your mood has been improving and stress levels are declining. Keep up the positive momentum!",
        "stable_wellbeing": "Your mental health appears stable. Keep monitoring your wellbeing regularly.",
        "mild_concern": "Your stress levels are showing some elevation. Consider using relaxation techniques from the app.",
        "increasing_stress": "Your stress levels are rising. Try the breathing exercises or chat with our support bot for help.",
        "high_stress": "Your stress levels are quite elevated. Please consider using the chatbot for support and try calming exercises.",
    }

    # Return appropriate template
    return templates.get(
        state, "Continue monitoring your mental health regularly with daily check-ins."
    )
