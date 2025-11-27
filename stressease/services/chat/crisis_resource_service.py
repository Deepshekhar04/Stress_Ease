"""
Crisis resources caching service.

This module handles caching and retrieval of country-specific
crisis resources in Firestore.
"""

from datetime import datetime
from typing import Dict, Optional, Any
from stressease.services.utility.firebase_config import get_firestore_client


# ============================================================================
# CRISIS RESOURCES OPERATIONS
# ============================================================================


def get_cached_crisis_resources(country: str) -> Optional[Dict[str, Any]]:
    """
    Get cached crisis resources for a specific country.
    Works with both country codes (e.g., 'US') and country names (e.g., 'United States').

    Args:
        country (str): Country code or name to get resources for

    Returns:
        dict: Crisis resources data, or None if not found in cache
    """
    db = get_firestore_client()

    if not country or not country.strip():
        print(
            "Warning: Attempted to get cached crisis resources with an empty country parameter."
        )
        return None

    try:
        # Normalize country input (uppercase for codes, title case for names)
        country_id = country.strip()
        if len(country_id) <= 3:  # Likely a country code
            country_id = country_id.upper()
        else:  # Likely a country name
            country_id = country_id.title()

        # Try to get document with country code/name as ID
        doc_ref = db.collection("crisis_resources").document(country_id)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()

        # If not found by exact match and it's a country name, try to find by country field
        if len(country_id) > 3:
            query = db.collection("crisis_resources").where("country", "==", country_id)
            docs = query.stream()
            for doc in docs:
                return doc.to_dict()

        return None

    except Exception as e:
        print(f"Error getting cached crisis resources for {country}: {str(e)}")
        return None


def cache_crisis_resources(country: str, resources: Dict[str, Any]) -> bool:
    """
    Cache crisis resources for a specific country.

    Args:
        country (str): Country code or name to cache resources for
        resources (dict): Crisis resources data to cache

    Returns:
        bool: True if successful, False otherwise
    """
    db = get_firestore_client()

    if not country or not country.strip():
        print(
            "Warning: Attempted to cache crisis resources with an empty country parameter."
        )
        return False

    try:
        # Normalize country input (uppercase for codes, title case for names)
        country_id = country.strip()
        if len(country_id) <= 3:  # Likely a country code
            country_id = country_id.upper()
        else:  # Likely a country name
            country_id = country_id.title()

        # Add country field to resources for querying
        resources["country"] = country_id
        resources["cached_at"] = datetime.utcnow()

        # Save to crisis_resources collection with country as document ID
        db.collection("crisis_resources").document(country_id).set(resources)
        return True

    except Exception as e:
        print(f"Error caching crisis resources for {country}: {str(e)}")
        return False
