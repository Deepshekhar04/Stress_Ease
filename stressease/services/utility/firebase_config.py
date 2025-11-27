"""
Firebase configuration and initialization.

This module handles Firebase Admin SDK setup and provides
the Firestore client instance to other services.
"""

import firebase_admin
from firebase_admin import credentials, firestore


# Global Firestore client
db = None


def init_firebase(credentials_path: str) -> None:
    """
    Initialize Firebase Admin SDK with service account credentials.
    
    Args:
        credentials_path (str): Path to the Firebase service account JSON file
        
    Raises:
        Exception: If Firebase initialization fails
    """
    global db
    
    try:
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        
        # Get Firestore client
        db = firestore.client()
        
        print("✓ Firebase Admin SDK initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize Firebase: {str(e)}")
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
        raise RuntimeError("Firebase has not been initialized. Call init_firebase() first.")
    return db
