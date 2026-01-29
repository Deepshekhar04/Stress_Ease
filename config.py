"""Configuration and environment loading."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """App configuration loaded from environment variables."""

    # Flask Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # Google Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # SerpApi Configuration (for SOS emergency contacts search)
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

    # Firebase Configuration (for local development only)
    # Cloud deployment uses FIREBASE_CREDENTIALS_JSON instead
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")

    @classmethod
    def validate_config(cls):
        """Validate required environment variables.

        Note: Firebase credentials are validated separately by firebase_config.py
        which supports both FIREBASE_CREDENTIALS_JSON (cloud) and
        FIREBASE_CREDENTIALS_PATH (local).
        """
        required_vars = [
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
        ]

        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please check your .env file or environment configuration."
            )

        # Production-specific validation
        if not cls.DEBUG:
            if cls.SECRET_KEY == "dev-secret-key-change-in-production":
                raise ValueError(
                    "SECRET_KEY must be changed from default value in production. "
                    "Set a secure SECRET_KEY in your environment variables."
                )

    @classmethod
    def setup_logging(cls):
        """Set up application logging based on DEBUG flag."""
        from stressease.services.utility.logger import setup_logging

        setup_logging(debug=cls.DEBUG)
