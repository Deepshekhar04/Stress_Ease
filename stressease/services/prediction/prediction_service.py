"""Stress prediction service using Gemini LLM with fallback logic."""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from stressease.services.chat.llm_service import base_llm


# ============================================================================
# PYDANTIC MODEL FOR STRUCTURED OUTPUT
# ============================================================================


class StressPrediction(BaseModel):
    """Structured stress prediction output from LLM."""

    stress_probability: float = Field(
        description="Probability of stress tomorrow (0.0 to 1.0), e.g., 0.76 for 76% chance"
    )
    label: str = Field(
        description="Stress level label: 'High' (>=0.7), 'Medium' (0.4-0.69), or 'Low' (<0.4)"
    )
    confidence: float = Field(
        description="Model confidence in this prediction (0.0 to 1.0), e.g., 0.73 for 73% confident"
    )


# ============================================================================
# PREDICTION LOGIC
# ============================================================================


def predict_stress(
    avg_mood_score: float, chat_count: int, avg_quiz_score: int
) -> Dict[str, Any]:
    """
    Predict tomorrow's stress using Gemini LLM with deterministic fallback.

    Args:
        avg_mood_score (float): 7-day average mood score (1.0 - 5.0)
        chat_count (int): Number of chat sessions in last 7 days (0 - 999)
        avg_quiz_score (int): Average sum of all 12 quiz questions over 7 days (12 - 60)

    Returns:
        Dict containing:
            - date: Tomorrow's date (YYYY-MM-DD)
            - stressProbability: Float (0-1)
            - label: "High", "Medium", or "Low"
            - confidence: Float (0-1)
            - basedOn: Echo of input data
    """
    # Calculate tomorrow's date
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Try LLM-based prediction first
    try:
        llm_result = _predict_with_llm(avg_mood_score, chat_count, avg_quiz_score)
        if llm_result:
            return {
                "date": tomorrow,
                "stressProbability": llm_result["stress_probability"],
                "label": llm_result["label"],
                "confidence": llm_result["confidence"],
                "basedOn": {
                    "avgMoodScore": avg_mood_score,
                    "chatCount": chat_count,
                    "avgQuizScore": avg_quiz_score,
                },
            }
    except Exception as e:
        print(f"⚠ LLM prediction failed, using fallback: {str(e)}")

    # Fallback to deterministic calculation
    fallback_result = _fallback_prediction(avg_mood_score, chat_count, avg_quiz_score)

    return {
        "date": tomorrow,
        "stressProbability": fallback_result["stress_probability"],
        "label": fallback_result["label"],
        "confidence": fallback_result["confidence"],
        "basedOn": {
            "avgMoodScore": avg_mood_score,
            "chatCount": chat_count,
            "avgQuizScore": avg_quiz_score,
        },
    }


def _predict_with_llm(
    avg_mood_score: float, chat_count: int, avg_quiz_score: int
) -> Optional[Dict[str, Any]]:
    """
    Use Gemini LLM to predict stress probability.

    Args:
        avg_mood_score: 7-day average mood score (1.0 - 5.0)
        chat_count: Number of chat sessions in last 7 days
        avg_quiz_score: Average sum of quiz questions over 7 days (12 - 60)

    Returns:
        Optional[Dict]: Prediction dict or None if LLM fails
    """
    if base_llm is None:
        raise RuntimeError("Gemini models not initialized. Call init_gemini() first.")

    try:
        # Create Pydantic output parser
        parser = PydanticOutputParser(pydantic_object=StressPrediction)

        # Build prompt template
        prompt_template = PromptTemplate(
            template="""You are an AI assistant analyzing mental health metrics to predict stress levels.

Given the following 7-day averages:
- Average Mood Score: {avg_mood_score}/5.0 (where 1=Very Poor, 5=Excellent)
- Chat Sessions Count: {chat_count} (seeking support/venting sessions)
- Average Quiz Score: {avg_quiz_score}/60 (sum of 12 daily questions, each 1-5)

**Prediction Task:**
Predict the probability that the user will experience HIGH stress tomorrow.

**Guidelines:**
- Lower mood scores (1-2) suggest higher stress risk
- Higher chat count suggests user is struggling/seeking more support
- Lower quiz scores (12-30) indicate poor overall wellness, higher stress risk
- Stress probability should be between 0.0 and 1.0
- Label: "High" if probability >= 0.7, "Medium" if 0.4-0.69, "Low" if < 0.4
- Confidence: How certain you are about this prediction (0.0 to 1.0)

{format_instructions}

Analyze and predict:""",
            input_variables=["avg_mood_score", "chat_count", "avg_quiz_score"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Chain: Prompt → Base LLM → Parser
        chain = prompt_template | base_llm | parser

        # Execute chain
        prediction_model = chain.invoke(
            {
                "avg_mood_score": avg_mood_score,
                "chat_count": chat_count,
                "avg_quiz_score": avg_quiz_score,
            }
        )

        # Convert Pydantic model to dict
        result = prediction_model.dict()

        # Validate and clamp values
        result["stress_probability"] = max(0.0, min(1.0, result["stress_probability"]))
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        # Ensure label matches probability
        prob = result["stress_probability"]
        if prob >= 0.7:
            result["label"] = "High"
        elif prob >= 0.4:
            result["label"] = "Medium"
        else:
            result["label"] = "Low"

        print(f"✓ LLM prediction successful: {result['label']} ({prob:.2f})")
        return result

    except Exception as e:
        print(f"✗ Error in LLM prediction: {str(e)}")
        return None


def _fallback_prediction(
    avg_mood_score: float, chat_count: int, avg_quiz_score: int
) -> Dict[str, Any]:
    """
    Deterministic fallback calculation when LLM is unavailable.

    Formula logic:
    - Mood score: Lower mood = higher stress (inverse relationship)
    - Chat count: More chats = more stress (linear relationship)
    - Quiz score: Lower quiz = higher stress (inverse relationship)

    Args:
        avg_mood_score: 7-day average mood score (1.0 - 5.0)
        chat_count: Number of chat sessions in last 7 days
        avg_quiz_score: Average sum of quiz questions over 7 days (12 - 60)

    Returns:
        Dict with stress_probability, label, and confidence
    """
    # Normalize inputs to 0-1 scale
    # Mood: Invert so lower mood = higher stress
    mood_factor = (5.0 - avg_mood_score) / 4.0  # Range: 0 (mood=5) to 1 (mood=1)

    # Chat count: Normalize with cap at 15 chats (more than 15 = max stress indicator)
    chat_factor = min(chat_count / 15.0, 1.0)  # Range: 0 to 1

    # Quiz score: Invert so lower quiz = higher stress
    quiz_factor = (60 - avg_quiz_score) / 48.0  # Range: 0 (quiz=60) to 1 (quiz=12)

    # Weighted average (you can adjust weights as needed)
    # Mood and quiz are more important than chat count
    stress_probability = (mood_factor * 0.4) + (chat_factor * 0.2) + (quiz_factor * 0.4)

    # Clamp to 0-1 range
    stress_probability = max(0.0, min(1.0, stress_probability))

    # Determine label
    if stress_probability >= 0.7:
        label = "High"
    elif stress_probability >= 0.4:
        label = "Medium"
    else:
        label = "Low"

    # Fallback has lower confidence than LLM predictions
    confidence = 0.65

    print(
        f"✓ Fallback prediction: {label} ({stress_probability:.2f}) - confidence: {confidence}"
    )

    return {
        "stress_probability": round(stress_probability, 2),
        "label": label,
        "confidence": confidence,
    }
