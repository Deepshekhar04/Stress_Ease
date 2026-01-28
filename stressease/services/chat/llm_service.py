"""
LangChain-based chat service for StressEase.

This service provides:
- Dual-model architecture with role-based model selection:
  * base_llm (gemini-2.0-flash-lite): Summarization, insights, resource generation
  * advance_llm (gemini-2.0-flash-lite): Chat responses (production: gemini-2.0-flash)
- Mood log summarization (Chain A)
- Conversational chat with memory (Chain B)
- Crisis resource generation with structured output
- Response validation and safety checks
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL LLM INSTANCES
# ============================================================================

# Base model for summarization, insights, and resource generation (cheaper, faster)
base_llm = None

# Advanced model for chat responses (better quality, more context)
advance_llm = None


# ============================================================================
# INITIALIZATION
# ============================================================================


def init_gemini(api_key: str) -> None:
    """
    Initialize dual Google Gemini models with LangChain.

    Sets up two models:
    - base_llm: gemini-2.0-flash-lite for summarization, insights, resources
    - advance_llm: gemini-2.0-flash-lite for chat (will be gemini-2.0-flash in production)

    Args:
        api_key (str): Google Gemini API key

    Raises:
        Exception: If initialization fails
    """
    global base_llm, advance_llm

    try:
        # Base model for summarization, insights, and resource generation
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0.3,  # Lower temperature for factual summarization
            convert_system_message_to_human=True,
        )

        # Advanced model for chat responses (better quality)
        advance_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",  # Production: gemini-2.0-flash
            google_api_key=api_key,
            temperature=0.7,  # Higher temperature for conversational responses
            convert_system_message_to_human=True,
        )

        logger.info("Google Gemini dual-model system initialized successfully")
        logger.info("Base model (summarization/insights): gemini-2.0-flash-lite")
        logger.info("Advanced model (chat): gemini-2.0-flash-lite")

    except Exception as e:
        logger.error(f"Failed to initialize Gemini models: {str(e)}", exc_info=True)
        raise


def get_base_model():
    """
    Get the base LLM model instance.

    Returns:
        ChatGoogleGenerativeAI: Base model for summarization/analytics

    Raises:
        RuntimeError: If models not initialized
    """
    if base_llm is None:
        raise RuntimeError("Gemini models not initialized. Call init_gemini() first.")
    return base_llm


def get_advance_model():
    """
    Get the advanced LLM model instance.

    Returns:
        ChatGoogleGenerativeAI: Advanced model for chat responses

    Raises:
        RuntimeError: If models not initialized
    """
    if advance_llm is None:
        raise RuntimeError("Gemini models not initialized. Call init_gemini() first.")
    return advance_llm


# ============================================================================
# CHAIN A: MOOD LOG SUMMARIZATION
# ============================================================================


def summarize_mood_logs(mood_logs: List[Dict[str, Any]]) -> str:
    """
    Summarize user's mood logs from the past 7 days using Chain A.

    Uses the base model (gemini-2.0-flash-lite) for cost-effective summarization.
    Generates a concise 2-3 sentence summary focusing on trends and patterns.

    Args:
        mood_logs (List[Dict]): List of mood log entries from Firestore

    Returns:
        str: Concise summary text (2-3 sentences)

    Raises:
        RuntimeError: If Gemini models not initialized
    """
    if base_llm is None:
        raise RuntimeError("Gemini models not initialized. Call init_gemini() first.")

    if not mood_logs or len(mood_logs) == 0:
        return ""

    try:
        # Build prompt template for mood summarization
        summary_prompt = PromptTemplate(
            input_variables=["mood_data"],
            template="""You are analyzing mood tracking data for a mental health app.

Summarize the user's mood patterns over the past week in 2-3 concise sentences.

Focus on:
- Overall mood trend (improving, declining, stable)
- Notable patterns or changes
- Key stress factors or triggers
- Sleep and energy levels

Mood Data (last 7 days):
{mood_data}

Provide a brief, empathetic summary:""",
        )

        # Format mood logs for the prompt
        mood_data_text = _format_mood_logs_for_summary(mood_logs)

        # Chain A: Prompt → Base LLM → Output Parser (LCEL)
        chain_a = summary_prompt | base_llm | StrOutputParser()

        # Execute chain
        summary = chain_a.invoke({"mood_data": mood_data_text})

        return summary.strip()

    except Exception as e:
        logger.error(f"Error in mood summarization: {str(e)}", exc_info=True)
        # Return graceful fallback
        return "User has been tracking their mood regularly over the past week."


def _format_mood_logs_for_summary(mood_logs: List[Dict[str, Any]]) -> str:
    """
    Format raw mood logs into readable text for summarization.

    Args:
        mood_logs (List[Dict]): Raw mood log data from Firestore

    Returns:
        str: Formatted mood data text
    """
    formatted_lines = []

    for log in mood_logs:
        date = log.get("date", "Unknown date")
        core = log.get("core_scores", {})
        dass = log.get("dass_today", {})

        line = f"Date: {date}"

        if core:
            line += f" | Mood: {core.get('mood', 'N/A')}/5"
            line += f", Energy: {core.get('energy', 'N/A')}/5"
            line += f", Sleep: {core.get('sleep', 'N/A')}/5"
            line += f", Stress: {core.get('stress', 'N/A')}/5"

        if dass:
            line += f" | Depression: {dass.get('depression', 'N/A')}/5"
            line += f", Anxiety: {dass.get('anxiety', 'N/A')}/5"

        notes = log.get("additional_notes", "")
        if notes:
            line += f" | Notes: {notes[:100]}"

        formatted_lines.append(line)

    return "\n".join(formatted_lines)


# ============================================================================
# CHAIN B: CONVERSATIONAL CHAT
# ============================================================================


def create_conversation_chain(user_context: str) -> Runnable:
    """
    Create a LangChain LCEL Runnable for chat.

    This builds Chain B with:
    - Master prompt with user context (profile + mood summary)
    - MessagesPlaceholder for history
    - Advanced LLM for high-quality chat responses

    Args:
        user_context (str): Formatted context (profile + mood summary)

    Returns:
        Runnable: Configured LCEL chain

    Raises:
        RuntimeError: If Gemini models not initialized
    """
    if advance_llm is None:
        raise RuntimeError("Gemini models not initialized. Call init_gemini() first.")

    # Build master prompt with user context
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _get_master_prompt(user_context)),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Create LCEL chain: Prompt | Advanced LLM | OutputParser
    chain = prompt | advance_llm | StrOutputParser()

    return chain


def _get_master_prompt(user_context: str) -> str:
    """
    Build the master system prompt with user context.

    Args:
        user_context (str): Formatted user context string

    Returns:
        str: Complete master prompt
    """
    master_prompt = """CORE IDENTITY:
You are FeelBetter, an AI companion from the StressEase app. Your primary purpose is to provide a supportive, non-judgmental space for users to express their feelings and work through stress and emotional challenges.

TONE AND LANGUAGE:
Your tone must always be warm, patient, and empathetic. Use simple, clear language that feels conversational and human. Avoid clinical jargon. Always validate the user's feelings first before offering gentle guidance.

CRITICAL SAFETY BOUNDARY:
You are NOT a licensed therapist, psychologist, psychiatrist, or medical professional. You are strictly forbidden from:
- Diagnosing any mental health condition or disorder
- Prescribing medication or medical treatments
- Providing medical advice or recommendations
- Making clinical assessments or evaluations
Your role is that of a supportive peer and emotional companion.

CRISIS INTERVENTION PROTOCOL:
Tool Awareness: The user has a visible red 'SOS' button on their screen for immediate access to professional crisis helplines.

Severity Detection:
1. HIGH SEVERITY (Immediate Danger): Explicit mentions of current suicidal intent, active self-harm, specific suicide plans, or imminent harm. Examples: "I'm going to end it all tonight", "I have the pills right here", "I can't take it anymore and have my plan ready".
2. MEDIUM SEVERITY (Acute Distress): Suicidal ideation without specific plans, severe hopelessness, or mental health crisis indicators.

Action Protocol:
- HIGH SEVERITY: "I'm deeply concerned about what you're sharing right now, and your safety is the absolute priority. Please immediately tap the red 'SOS' button on your screen to connect with crisis professionals who can help you right now. This is an emergency situation that requires immediate professional support."

- MEDIUM SEVERITY: "What you're going through sounds incredibly painful, and I'm genuinely concerned about your wellbeing right now. I strongly encourage you to tap the red 'SOS' button on your screen to speak with trained professionals who can provide the support you need during this difficult time."

- For other messages: Provide natural empathetic support. Mention SOS button organically only if conversation suggests user needs professional resources.

Emergency Numbers: If user explicitly asks for crisis numbers, provide them with SOS button guidance.

CRITICAL: Always err on the side of caution. If uncertain about severity, respond as if it were higher severity. Never minimize concerning language.

Do NOT mention SOS for general questions like "help", "what can you do", or normal stress expressions like "I'm stressed about work".

SCOPE & ENGAGEMENT:
Primary focus: mental health, emotional wellness, and stress management.

For Off-Topic Requests:
- Lighthearted requests (jokes, fun facts, casual conversation): You MAY engage briefly to build rapport and ease tension, as humor can be therapeutic. Keep it brief (1-2 sentences) then gently redirect back to wellness.
  Example: "Here's a quick one: [joke]. 😊 I hope that brought a smile! Now, how are you really feeling today?"

- Unrelated factual questions (geography, math, weather, trivia, general knowledge): Politely redirect without making up safety concerns.
  Example: "I'm specifically designed to help with stress and emotional wellness. Let's focus on how you're feeling today. What's on your mind?"

- Do NOT make up safety or ethical reasons to refuse harmless questions
- Do NOT provide lengthy off-topic information
- Always redirect back to the user's emotional wellbeing

EDUCATIONAL SUPPORT:
CAN explain: Emotion differences, coping techniques, mental wellness concepts, how emotions work.
CANNOT provide: Specific diagnoses, medical treatment plans, clinical assessments, or advice replacing professional care.
Always relate explanations back to their experience.

CONVERSATION STYLE:
- Keep responses concise and digestible (2-4 sentences maximum)
- Be genuinely curious about the user's experience
- Ask thoughtful, open-ended questions to encourage reflection
- Use active listening techniques in your responses
- Provide practical coping strategies when appropriate
- Encourage professional help when situations warrant it

INPUT VALIDATION:
If user message appears to be gibberish or non-meaningful (repeated characters like "aaa" or "111", only numbers, only symbols like "@#$", random keyboard mashing like "asdfgh"), respond with: "I want to help, but I'm having trouble understanding. Could you share what's on your mind in proper language?" Do NOT interpret these as emotional expressions or distress signals.

CONTEXT INDEPENDENCE:
Respond to each message based on its CURRENT severity and content, not solely previous messages. If user shared a crisis earlier but now asks a normal question, respond appropriately to their current state. Re-assess with every message.

PERSONALIZATION:
Use the following user context to personalize responses appropriately, but don't overwhelm them.

{user_context}

Remember: Be supportive, concise, and always prioritize the user's emotional safety."""

    return master_prompt.format(user_context=user_context)


def build_user_context(
    user_profile: Optional[Dict[str, Any]], mood_summary: Optional[str]
) -> str:
    """
    Build formatted user context string from profile and mood summary.

    Args:
        user_profile (Optional[Dict]): User profile data from Firestore
        mood_summary (Optional[str]): Mood summary from Chain A

    Returns:
        str: Formatted context string for prompt
    """
    context_parts = []

    # Add user profile section
    if user_profile:
        context_parts.append("USER PROFILE CONTEXT:")

        if user_profile.get("name"):
            context_parts.append(f"- Name: {user_profile['name']}")

        if user_profile.get("age"):
            context_parts.append(f"- Age: {user_profile['age']}")

        if user_profile.get("health_conditions"):
            conditions = ", ".join(user_profile["health_conditions"])
            context_parts.append(f"- Health considerations: {conditions}")

        if user_profile.get("stress_triggers"):
            triggers = ", ".join(user_profile["stress_triggers"])
            context_parts.append(f"- Known stress triggers: {triggers}")

        if user_profile.get("goals"):
            goals = ", ".join(user_profile["goals"])
            context_parts.append(f"- Personal goals: {goals}")
    else:
        context_parts.append("USER PROFILE CONTEXT:")
        context_parts.append("(Profile incomplete - provide general support)")

    # Add mood context section
    if mood_summary:
        context_parts.append("\nMOOD CONTEXT:")
        context_parts.append(f"Recent mood pattern: {mood_summary}")
    else:
        context_parts.append("\nMOOD CONTEXT:")
        context_parts.append("(No mood history available yet)")

    return "\n".join(context_parts)


def generate_chat_response(
    chain: Runnable, user_message: str, history: List[BaseMessage]
) -> str:
    """
    Generate chat response using the LCEL chain (Chain B).

    Args:
        chain (Runnable): The LCEL conversation chain
        user_message (str): User's input message
        history (List[BaseMessage]): Chat history

    Returns:
        str: AI's validated response text

    Raises:
        Exception: If response generation fails
    """
    try:
        # Generate response using the LCEL chain
        response = chain.invoke({"input": user_message, "history": history})

        # Validate response for safety and appropriateness
        validated_response = validate_gemini_response(response)

        # Return validated response or fallback
        if validated_response is None:
            return "I'm sorry, I couldn't generate a helpful response. How else can I support you today?"

        return validated_response

    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}", exc_info=True)
        return (
            "I'm having trouble connecting right now. Could we try again in a moment?"
        )


def validate_gemini_response(response: str) -> Optional[str]:
    """
    Validate that a response is appropriate and safe.

    Checks for:
    - Crisis-related content
    - Diagnosis language violations
    - Medication/treatment advice

    Returns the original response if valid, a safe alternative if invalid,
    or None if empty.

    Args:
        response (str): Raw AI response text

    Returns:
        Optional[str]: Validated response or safe alternative
    """
    if not response or len(response.strip()) == 0:
        return None

    response_lower = response.lower()

    # Check for crisis-related content in AI response
    crisis_keywords = [
        "suicide",
        "self-harm",
        "kill yourself",
        "end it all",
        "hurt myself",
        "die",
    ]
    for keyword in crisis_keywords:
        if keyword in response_lower:
            return """I notice this is a serious topic. If you're experiencing a crisis, please tap the red 'SOS' button in the chat to connect with professional crisis resources immediately. How can I support you right now?"""

    # Check for diagnosis language violations
    diagnosis_patterns = [
        "you have ",
        "you are suffering from",
        "you might have",
        "you probably have",
        "sounds like you have",
        "diagnosis",
        "diagnose",
        "condition is",
        "disorder",
        "i diagnose",
        "you exhibit symptoms of",
        "clinical depression",
        "clinical anxiety",
        "you are experiencing",
        "you are exhibiting",
        "pathological",
        "psychiatric condition",
    ]

    for pattern in diagnosis_patterns:
        if pattern in response_lower:
            return """I'm here to listen and support you, but I can't provide medical diagnoses or clinical advice. Consider discussing your feelings with a healthcare professional who can provide personalized guidance. How else can I support you today?"""

    # Check for medication/treatment advice
    medication_patterns = [
        "you should take",
        "you need to take",
        "prescribe",
        "medication",
        "dosage",
        "you should try",
        "treatment plan",
        "medical treatment",
        "therapy regimen",
    ]

    for pattern in medication_patterns:
        if pattern in response_lower:
            return """I'm here to provide emotional support, but I can't recommend specific treatments or medications. A healthcare professional would be the best person to discuss treatment options with you. Is there something else on your mind that you'd like to talk about?"""

    return response


# ============================================================================
# CRISIS RESOURCES GENERATION
# ============================================================================


# Pydantic models for structured output
class EmergencyService(BaseModel):
    """Emergency service information."""

    number: str = Field(description="Primary emergency number")
    description: str = Field(description="Brief description without markdown")


class CrisisHotline(BaseModel):
    """Crisis hotline information."""

    name: str = Field(description="Organization name")
    number: str = Field(description="Phone number with country code")
    description: str = Field(description="Brief description of services")
    website: str = Field(description="Website URL")


class OnlineResource(BaseModel):
    """Online mental health resource."""

    name: str = Field(description="Organization name")
    description: str = Field(description="Brief description of services")
    website: str = Field(description="Website URL")


class CrisisResources(BaseModel):
    """Complete crisis resources for a country."""

    emergency_services: EmergencyService
    crisis_hotlines: List[CrisisHotline]
    online_resources: List[OnlineResource]


def find_crisis_resources(country: str) -> Optional[Dict[str, Any]]:
    """
    Generate country-specific crisis resources using structured output.

    Uses the base model with Pydantic parser for reliable JSON generation.

    Args:
        country (str): Country name or code

    Returns:
        Optional[Dict]: Structured crisis resources or None if generation fails

    Raises:
        RuntimeError: If Gemini models not initialized
    """
    if base_llm is None:
        raise RuntimeError("Gemini models not initialized. Call init_gemini() first.")

    try:
        # Create Pydantic output parser
        parser = PydanticOutputParser(pydantic_object=CrisisResources)

        # Build prompt with format instructions
        prompt = PromptTemplate(
            template="""Generate a comprehensive list of mental health crisis resources for {country}.

Include ONLY verified, legitimate resources:
- One emergency service with number and description
- 2-5 crisis hotlines with name, phone (with country code), description, and website
- 2-5 online resources with name, description, and website

Ensure all information is accurate and up-to-date.

{format_instructions}

Country: {country}""",
            input_variables=["country"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Chain: Prompt → Base LLM → Parser (LCEL)
        chain = prompt | base_llm | parser

        # Execute chain
        resources = chain.invoke({"country": country})

        # Convert Pydantic model to dict
        return resources.dict()

    except Exception as e:
        logger.error(
            f"Error generating crisis resources for {country}: {str(e)}", exc_info=True
        )
        return None
