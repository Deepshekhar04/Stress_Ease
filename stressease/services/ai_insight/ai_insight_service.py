"""
AI insights and suggestions service for daily mood quiz analysis.

This service generates personalized mental health recommendations based on
current day's mood quiz data using Google Gemini LLM.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from stressease.services.utility.firebase_config import get_firestore_client
from stressease.services.chat.llm_service import base_llm


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================


class AIInsights(BaseModel):
    """Structured AI insights output for daily mood quiz."""

    dominant_emotion: str = Field(
        description="Dominant emotion based on today's scores: Happy/Neutral/Sad/Anxious/Stressed/Energetic/Calm/Tired"
    )
    summary: str = Field(
        description="2-3 sentence summary of today's mood state and key observations"
    )
    motivation_quote: str = Field(
        description="Short motivational quote with emoji, personalized to today's mood"
    )
    suggestions: List[str] = Field(
        description="3-5 actionable suggestions for today/tomorrow, specific to detected issues"
    )


# ============================================================================
# MAIN INSIGHTS GENERATION
# ============================================================================


def generate_ai_insights(
    user_id: str, daily_quiz_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Main orchestrator function that generates AI insights from daily quiz data.

    Args:
        user_id (str): Firebase Auth user ID
        daily_quiz_data (dict): Current day's quiz data with core_scores, dass_today, etc.

    Returns:
        Optional[Dict]: Structured insights data or None if generation fails
    """
    try:
        # Analyze current day's mood with LLM
        insights = analyze_daily_mood(daily_quiz_data)

        if not insights:
            print(f"⚠ LLM returned no insights for user {user_id}")
            return None

        # Validate structure
        if not _validate_insights_structure(insights):
            print(f"⚠ Invalid insights structure for user {user_id}")
            return None

        # Save to Firestore
        success = save_ai_insights_to_firestore(user_id, insights)

        if not success:
            print(f"⚠ Failed to save insights to Firestore for user {user_id}")
            return None

        return insights

    except Exception as e:
        print(f"✗ Error generating AI insights for user {user_id}: {str(e)}")
        return None


def analyze_daily_mood(quiz_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Analyze current day's mood quiz data using Google Gemini LLM.

    Uses structured output with Pydantic parser for reliable JSON generation.

    Args:
        quiz_data (dict): Daily quiz data (core_scores, dass_today, rotating_scores, etc.)

    Returns:
        Optional[Dict]: Insights dictionary or None if analysis fails
    """
    if base_llm is None:
        raise RuntimeError("Gemini models not initialized. Call init_gemini() first.")

    try:
        # Create Pydantic output parser
        parser = PydanticOutputParser(pydantic_object=AIInsights)

        # Build prompt with daily quiz data
        prompt_template = PromptTemplate(
            template="""You are a compassionate AI mental health assistant analyzing daily mood quiz data.

Based on today's mood quiz scores, provide personalized insights and suggestions.

**Today's Mood Data:**
{mood_data}

**Score Interpretation:**
- 1 = Very Poor/Very Low
- 2 = Poor/Low
- 3 = Moderate/Average
- 4 = Good/High
- 5 = Excellent/Very High

**Instructions:**
1. **Dominant Emotion**: Choose the most fitting emotion based on all scores (Happy/Neutral/Sad/Anxious/Stressed/Energetic/Calm/Tired)
2. **Summary**: Write 2-3 sentences about today's mood state, highlighting key observations
3. **Motivation Quote**: Create a short, encouraging quote with emoji that resonates with today's mood
4. **Suggestions**: Provide 3-5 specific, actionable suggestions for today/tomorrow addressing detected issues

**Tone**: Empathetic, supportive, non-clinical. Avoid medical terminology or diagnosis.

{format_instructions}

Generate insights:""",
            input_variables=["mood_data"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Format the quiz data for the prompt
        mood_data_text = _build_daily_prompt(quiz_data)

        # Chain: Prompt → Base LLM → Parser (LCEL)
        chain = prompt_template | base_llm | parser

        # Execute chain
        insights_model = chain.invoke({"mood_data": mood_data_text})

        # Convert Pydantic model to dict
        return insights_model.dict()

    except Exception as e:
        print(f"Error analyzing daily mood: {str(e)}")
        return None


def save_ai_insights_to_firestore(user_id: str, insights: Dict[str, Any]) -> bool:
    """
    Write AI insights to Firestore at users/{userId}/ai_insights/latest.

    Args:
        user_id (str): Firebase Auth user ID
        insights (dict): Insights data to save

    Returns:
        bool: True if successful, False otherwise
    """
    db = get_firestore_client()

    try:
        # Add metadata
        insights_doc = {
            "dominant_emotion": insights["dominant_emotion"],
            "summary": insights["summary"],
            "motivation_quote": insights["motivation_quote"],
            "suggestions": insights["suggestions"],
            "generated_at": datetime.utcnow(),
        }

        # Write to Firestore (overwrites previous insights)
        doc_ref = (
            db.collection("users")
            .document(user_id)
            .collection("ai_insights")
            .document("latest")
        )
        doc_ref.set(insights_doc)

        print(f"✓ AI insights saved to Firestore for user {user_id}")
        return True

    except Exception as e:
        print(f"✗ Error saving insights to Firestore for user {user_id}: {str(e)}")
        return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _build_daily_prompt(quiz_data: Dict[str, Any]) -> str:
    """
    Format daily quiz data into readable text for LLM prompt.

    Prioritizes enriched Q&A format when available for better context.

    Args:
        quiz_data (dict): Daily quiz data from mood.py

    Returns:
        str: Formatted mood data text
    """
    lines = []

    # Date
    date = quiz_data.get("date", "Unknown")
    lines.append(f"Date: {date}")

    # Check if we have enriched Q&A data
    enriched_qa = quiz_data.get("enriched_qa", [])

    if enriched_qa:
        # Use enriched Q&A format for better context
        lines.append("\n**Today's Quiz Responses (Question & Answer):**\n")

        # Group by dimension for better readability
        core_qa = [qa for qa in enriched_qa if qa.get("dimension") == "core"]
        rotating_qa = [qa for qa in enriched_qa if qa.get("dimension") == "rotating"]
        dass_qa = [qa for qa in enriched_qa if qa.get("dimension") == "dass"]

        if core_qa:
            lines.append("**Core Well-being:**")
            for qa in core_qa:
                score = qa.get("score", 0)
                question = qa.get("question", "")
                score_label = _get_score_label(score)
                lines.append(f"  Q: {question}")
                lines.append(f"  A: {score}/5 ({score_label})")
                lines.append("")

        if rotating_qa:
            domain = rotating_qa[0].get("domain", "Life Area")
            lines.append(f"**{domain.capitalize()} Domain:**")
            for qa in rotating_qa:
                score = qa.get("score", 0)
                question = qa.get("question", "")
                score_label = _get_score_label(score)
                lines.append(f"  Q: {question}")
                lines.append(f"  A: {score}/5 ({score_label})")
                lines.append("")

        if dass_qa:
            lines.append("**Mental Health Indicators (DASS-21):**")
            for qa in dass_qa:
                score = qa.get("score", 0)
                question = qa.get("question", "")
                score_label = _get_score_label(score)
                lines.append(f"  Q: {question}")
                lines.append(f"  A: {score}/5 ({score_label})")
                lines.append("")
    else:
        # Fallback to original score-only format
        # Core scores
        core = quiz_data.get("core_scores", {})
        if core:
            lines.append(f"\nCore Metrics:")
            lines.append(f"  - Mood: {core.get('mood', 'N/A')}/5")
            lines.append(f"  - Energy: {core.get('energy', 'N/A')}/5")
            lines.append(f"  - Sleep Quality: {core.get('sleep', 'N/A')}/5")
            lines.append(f"  - Stress Level: {core.get('stress', 'N/A')}/5")

        # DASS scores
        dass = quiz_data.get("dass_today", {})
        if dass:
            lines.append(f"\nMental Health Indicators (DASS-21):")
            lines.append(f"  - Depression: {dass.get('depression', 'N/A')}/5")
            lines.append(f"  - Anxiety: {dass.get('anxiety', 'N/A')}/5")
            lines.append(f"  - Stress: {dass.get('stress', 'N/A')}/5")

        # Rotating domain
        rotating = quiz_data.get("rotating_scores", {})
        if rotating:
            domain_name = rotating.get("domain_name", "Unknown")
            scores = rotating.get("scores", [])
            if scores:
                avg_domain = sum(scores) / len(scores)
                lines.append(f"\n{domain_name.capitalize()} Domain:")
                lines.append(f"  - Average Score: {avg_domain:.1f}/5")
                lines.append(f"  - Individual Scores: {scores}")

        # Calculated averages
        core_avg = quiz_data.get("core_avg")
        rotating_avg = quiz_data.get("rotating_avg")
        if core_avg is not None:
            lines.append(f"\nOverall Core Average: {core_avg:.2f}/5")
        if rotating_avg is not None:
            lines.append(
                f"Overall {rotating.get('domain_name', 'Domain')} Average: {rotating_avg:.2f}/5"
            )

    # Additional notes (always include if present)
    notes = quiz_data.get("additional_notes", "")
    if notes:
        lines.append(f"\nUser Notes: {notes}")

    return "\n".join(lines)


def _get_score_label(score: int) -> str:
    """Convert numeric score to descriptive label."""
    labels = {
        1: "Very Poor/Very Low",
        2: "Poor/Low",
        3: "Moderate/Average",
        4: "Good/High",
        5: "Excellent/Very High",
    }
    return labels.get(score, "Unknown")


def _validate_insights_structure(insights: Dict[str, Any]) -> bool:
    """
    Validate that insights contain all required fields with correct types.

    Args:
        insights (dict): Insights dictionary to validate

    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = {
        "dominant_emotion": str,
        "summary": str,
        "motivation_quote": str,
        "suggestions": list,
    }

    for field, expected_type in required_fields.items():
        if field not in insights:
            print(f"⚠ Missing required field: {field}")
            return False

        if not isinstance(insights[field], expected_type):
            print(
                f"⚠ Invalid type for {field}: expected {expected_type}, got {type(insights[field])}"
            )
            return False

    # Validate suggestions is a list of strings
    if not all(isinstance(s, str) for s in insights["suggestions"]):
        print(f"⚠ Suggestions must be a list of strings")
        return False

    return True
