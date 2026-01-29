"""
Firebase configuration and initialization.

This module handles Firebase Admin SDK setup and provides
the Firestore client instance to other services.
"""

import firebase_admin
from firebase_admin import credentials, firestore
import logging

logger = logging.getLogger(__name__)


# Global Firestore client
db = None


def init_firebase(credentials_path: str = None) -> None:
    """
    Initialize Firebase Admin SDK with service account credentials.

    Supports two methods:
    1. Environment variable (cloud deployment): FIREBASE_CREDENTIALS_JSON
    2. File path (local development): credentials_path parameter

    Args:
        credentials_path (str, optional): Path to Firebase service account JSON file

    Raises:
        Exception: If Firebase initialization fails
    """
    global db

    try:
        import os
        import json

        # Method 1: Try environment variable first (cloud deployment)
        firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

        if firebase_creds_json:
            # Parse JSON string from environment variable
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)
            logger.info(
                "Firebase initialized from FIREBASE_CREDENTIALS_JSON environment variable"
            )

        # Method 2: Fall back to file path (local development)
        elif credentials_path:
            cred = credentials.Certificate(credentials_path)
            logger.info(
                f"Firebase initialized from credentials file: {credentials_path}"
            )

        else:
            raise ValueError(
                "Firebase credentials not found. Provide either:\n"
                "  1. FIREBASE_CREDENTIALS_JSON environment variable (for cloud), or\n"
                "  2. FIREBASE_CREDENTIALS_PATH file path (for local development)"
            )

        # Initialize Firebase Admin SDK
        firebase_admin.initialize_app(cred)

        # Get Firestore client
        db = firestore.client()

        logger.info("Firebase Admin SDK initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}", exc_info=True)
        raise


def get_firestore_client():
    """
    Get the Firestore client instance.

    Returns:
        firestore.Client: The Firestore client

    Raises:
        RuntimeError: If Firebase hasn't been initialized
    """
    if db is None:
        raise RuntimeError(
            "Firebase has not been initialized. Call init_firebase() first."
        )
    return db
