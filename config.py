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

    # Firebase Configuration
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")

    # LangSmith Configuration (Monitoring)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "stressease-prod")

    @classmethod
    def validate_config(cls):
        """Validate required environment variables and credential path."""
        required_vars = [
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            ("FIREBASE_CREDENTIALS_PATH", cls.FIREBASE_CREDENTIALS_PATH),
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

        # Check if Firebase credentials file exists
        if not os.path.exists(cls.FIREBASE_CREDENTIALS_PATH):
            raise ValueError(
                f"Firebase credentials file not found at: {cls.FIREBASE_CREDENTIALS_PATH}. "
                "Please ensure the file exists and the path is correct."
            )
