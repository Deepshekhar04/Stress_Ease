"""
SOS Emergency Contacts Service.

This module handles fetching, verifying, and caching emergency and mental health
crisis contacts using SerpApi for real-time web searches and LLM for intelligent
data extraction and validation.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any, List
from serpapi import GoogleSearch
from config import Config
from stressease.services.chat.crisis_resource_service import (
    get_cached_crisis_resources,
    cache_crisis_resources,
)
from stressease.services.chat import llm_service


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_TTL_DAYS = 30
CONTACT_COUNT = 5


# ============================================================================
# MAIN SOS CONTACTS OPERATIONS
# ============================================================================


def get_sos_contacts(
    country: str, force_refresh: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Get SOS emergency contacts for a specific country.

    Implements smart caching:
    - If cache is fresh (< 30 days), return cached data
    - If cache is stale (>= 30 days) or missing, fetch fresh data
    - If force_refresh=True, bypass cache and fetch fresh

    Args:
        country (str): Country name (e.g., 'India', 'United States')
        force_refresh (bool): If True, bypass cache and fetch fresh data

    Returns:
        dict: SOS contacts data with crisis_hotlines array, or None if error
    """
    try:
        # Step 1: Check cache unless force refresh
        if not force_refresh:
            cached_data = get_cached_crisis_resources(country)

            if cached_data:
                # Validate cache freshness
                if is_cache_valid(cached_data, ttl_days=DEFAULT_TTL_DAYS):
                    print(f"✓ Returning fresh cached data for {country}")
                    return cached_data
                else:
                    print(f"⚠ Cache expired for {country}, fetching fresh data")

        # Step 2: Fetch fresh data using SerpApi + LLM
        print(f"🔍 Fetching fresh SOS contacts for {country}")
        fresh_contacts = fetch_fresh_sos_contacts(country)

        if fresh_contacts:
            # Step 3: Cache the fresh data
            cache_success = cache_crisis_resources(country, fresh_contacts)
            if cache_success:
                print(f"✓ Cached fresh contacts for {country}")
            return fresh_contacts

        # Step 4: Fallback to cached data even if stale
        print(f"⚠ Could not fetch fresh data, attempting to use stale cache")
        return get_cached_crisis_resources(country)

    except Exception as e:
        print(f"❌ Error in get_sos_contacts for {country}: {str(e)}")
        # Fallback to cache on any error
        return get_cached_crisis_resources(country)


def fetch_fresh_sos_contacts(country: str) -> Optional[Dict[str, Any]]:
    """
    Fetch fresh SOS contacts from web using SerpApi and LLM.

    Workflow:
    1. Search web using SerpApi for emergency contacts
    2. Extract and structure data using Gemini LLM
    3. Validate output has exactly 5 contacts
    4. Return structured contact data

    Args:
        country (str): Country name to fetch contacts for

    Returns:
        dict: Fresh SOS contacts data, or None if fetch fails
    """
    try:
        # Step 1: Search for emergency and crisis contacts
        search_results = search_emergency_contacts(country)

        if not search_results:
            print(f"⚠ No search results found for {country}")
            return None

        # Step 2: Extract contacts using LLM
        extracted_contacts = extract_contacts_with_llm(search_results, country)

        if not extracted_contacts:
            print(f"⚠ LLM failed to extract contacts for {country}")
            return None

        # Step 3: Validate structure
        if not validate_contact_structure(extracted_contacts):
            print(f"❌ Invalid contact structure for {country}")
            return None

        return extracted_contacts

    except Exception as e:
        print(f"❌ Error fetching fresh contacts for {country}: {str(e)}")
        return None


def search_emergency_contacts(country: str) -> List[Dict[str, Any]]:
    """
    Search for emergency and mental health crisis contacts using SerpApi.

    Executes multiple targeted searches:
    1. Emergency services (police, ambulance, national emergency)
    2. Mental health crisis hotlines
    3. Suicide prevention hotlines

    Args:
        country (str): Country name

    Returns:
        list: Combined search results from all queries
    """
    try:
        api_key = Config.SERPAPI_API_KEY
        if not api_key:
            print("❌ SerpApi API key not configured")
            return []

        all_results = []

        # Get dynamic current year
        current_year = datetime.now().year

        # Query 1: Emergency services
        query1 = f"{country} emergency number national emergency police ambulance {current_year}"
        results1 = execute_serpapi_search(query1, api_key)
        all_results.extend(results1)

        # Query 2: Mental health crisis hotlines
        query2 = f"{country} mental health crisis hotline suicide prevention {current_year} official"
        results2 = execute_serpapi_search(query2, api_key)
        all_results.extend(results2)

        # Query 3: Verified organizations
        query3 = f"{country} crisis helpline phone number website {current_year}"
        results3 = execute_serpapi_search(query3, api_key)
        all_results.extend(results3)

        print(f"✓ Retrieved {len(all_results)} search results for {country}")
        return all_results

    except Exception as e:
        print(f"❌ Error in SerpApi search for {country}: {str(e)}")
        return []


def execute_serpapi_search(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Execute a single SerpApi search query.

    Args:
        query (str): Search query
        api_key (str): SerpApi API key

    Returns:
        list: Organic search results
    """
    try:
        params = {
            "q": query,
            "api_key": api_key,
            "num": 10,  # Get top 10 results
            "hl": "en",  # Language: English
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        # Extract organic results
        organic_results = results.get("organic_results", [])
        return organic_results

    except Exception as e:
        print(f"⚠ SerpApi search error: {str(e)}")
        return []


def extract_contacts_with_llm(
    search_results: List[Dict[str, Any]], country: str
) -> Optional[Dict[str, Any]]:
    """
    Use Gemini LLM to extract and structure emergency contacts from search results.

    Uses the existing base_llm model (gemini-2.0-flash-lite) for cost-effective extraction.

    The LLM analyzes search results and extracts:
    - Exactly 5 contacts
    - First contact: National emergency number
    - Next 4: Top mental health crisis hotlines
    - Structured in the exact format required

    Args:
        search_results (list): Raw search results from SerpApi
        country (str): Country name for context

    Returns:
        dict: Structured contact data, or None if extraction fails
    """
    try:
        # Use existing base model (follows codebase architecture)
        base_llm = llm_service.get_base_model()

        # Prepare search results summary
        search_summary = prepare_search_summary(search_results)

        # Create prompt
        prompt = create_extraction_prompt(country, search_summary)

        # Generate response using LangChain
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt_template = PromptTemplate(
            template="{prompt}", input_variables=["prompt"]
        )

        # Chain: Prompt → Base LLM → Parser (following LCEL pattern)
        chain = prompt_template | base_llm | StrOutputParser()

        # Execute chain
        response = chain.invoke({"prompt": prompt})

        # Parse LLM response
        contacts_data = parse_llm_response(response, country)

        return contacts_data

    except Exception as e:
        print(f"❌ LLM extraction error for {country}: {str(e)}")
        return None


def create_extraction_prompt(country: str, search_summary: str) -> str:
    """
    Create the prompt for LLM contact extraction.

    Args:
        country (str): Country name
        search_summary (str): Summary of search results

    Returns:
        str: Formatted prompt for LLM
    """
    current_year = datetime.now().year

    prompt = f"""You are an emergency contact information specialist.

Task: Extract EXACTLY 5 emergency and crisis contacts for {country} from the search results below.

CRITICAL Requirements:
1. First contact MUST be the national emergency number (e.g., 112 for India, 911 for USA, 999 for UK)
2. Next 4 contacts MUST be mental health crisis hotlines (suicide prevention, crisis support)
3. Prioritize OFFICIAL sources (.gov, .org domains)
4. Verify all information is current ({current_year})
5. Extract EXACTLY 5 contacts - no more, no less

Output ONLY valid JSON in this EXACT format (no markdown, no code blocks):
{{
  "crisis_hotlines": [
    {{
      "name": "National Emergency Number",
      "number": "112",
      "website": "https://112.gov.in/",
      "description": "National emergency response system for all emergencies"
    }},
    {{
      "name": "Organization Name",
      "number": "+XX-XXXXXXXXXX",
      "website": "https://example.org/",
      "description": "Brief description of services"
    }}
  ]
}}

Search Results:
{search_summary}

Extract EXACTLY 5 contacts in valid JSON format:"""

    return prompt


def prepare_search_summary(search_results: List[Dict[str, Any]]) -> str:
    """
    Prepare a concise summary of search results for LLM.

    Args:
        search_results (list): Raw search results

    Returns:
        str: Formatted summary
    """
    summary_parts = []

    for idx, result in enumerate(search_results[:15], 1):  # Limit to top 15 results
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")

        summary_parts.append(f"{idx}. {title}\n   {snippet}\n   {link}\n")

    return "\n".join(summary_parts)


def parse_llm_response(response_text: str, country: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response and extract structured contact data.

    Args:
        response_text (str): Raw LLM response
        country (str): Country name

    Returns:
        dict: Structured contacts, or None if parsing fails
    """
    try:
        import json

        # Clean response (remove markdown code blocks if present)
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        # Parse JSON
        contacts_data = json.loads(cleaned_text)

        # Add country field
        contacts_data["country"] = country

        return contacts_data

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {str(e)}")
        print(f"Response text: {response_text[:200]}")
        return None
    except Exception as e:
        print(f"❌ Error parsing LLM response: {str(e)}")
        return None


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def is_cache_valid(cached_data: Dict[str, Any], ttl_days: int = 30) -> bool:
    """
    Check if cached SOS contacts are still valid.

    Args:
        cached_data (dict): Cached contact data
        ttl_days (int): Time-to-live in days (default: 30)

    Returns:
        bool: True if cache is valid (fresh), False if expired
    """
    try:
        cached_at = cached_data.get("cached_at")

        if not cached_at:
            print("⚠ No cached_at field found")
            return False

        # Handle both datetime objects and timestamps
        if isinstance(cached_at, datetime):
            cache_time = cached_at
        else:
            # Assume it's a Firestore timestamp
            cache_time = cached_at

        # Calculate age
        age = datetime.now(timezone.utc) - cache_time

        # Check if within TTL
        is_valid = age.days < ttl_days

        if is_valid:
            print(f"✓ Cache is fresh ({age.days} days old, TTL: {ttl_days} days)")
        else:
            print(f"⚠ Cache is stale ({age.days} days old, TTL: {ttl_days} days)")

        return is_valid

    except Exception as e:
        print(f"❌ Error validating cache: {str(e)}")
        return False


def validate_contact_structure(contacts: Dict[str, Any]) -> bool:
    """
    Validate that contact data has the correct structure.

    Requirements:
    - Must have "crisis_hotlines" array
    - Must have exactly 5 contacts
    - Each contact must have: name, number, website, description

    Args:
        contacts (dict): Contact data to validate

    Returns:
        bool: True if valid structure, False otherwise
    """
    try:
        # Check for crisis_hotlines array
        if "crisis_hotlines" not in contacts:
            print("❌ Missing crisis_hotlines array")
            return False

        hotlines = contacts["crisis_hotlines"]

        # Check count
        if len(hotlines) != CONTACT_COUNT:
            print(f"❌ Expected {CONTACT_COUNT} contacts, got {len(hotlines)}")
            return False

        # Validate each contact
        required_fields = ["name", "number", "website", "description"]
        for idx, contact in enumerate(hotlines, 1):
            for field in required_fields:
                if field not in contact or not contact[field]:
                    print(f"❌ Contact {idx} missing field: {field}")
                    return False

        print(f"✓ Contact structure valid ({CONTACT_COUNT} contacts)")
        return True

    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return False
