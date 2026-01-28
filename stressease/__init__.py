"""Flask app factory. Registers blueprints and initializes services."""

from flask import Flask, jsonify
from config import Config
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_app():
    """
    Application factory function that creates and configures the Flask app.

    Returns:
        Flask: Configured Flask application instance
    """
    # Create Flask application instance
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(Config)

    # Set up logging first
    Config.setup_logging()
    logger.info("Starting StressEase Backend API initialization")

    # Initialize services
    from stressease.services.utility.firebase_config import init_firebase
    from stressease.services.chat.llm_service import init_gemini

    try:
        # Initialize Firebase
        init_firebase(Config.FIREBASE_CREDENTIALS_PATH)
        logger.info("Firebase initialized successfully")

        # Initialize Gemini AI (dual-model LLM)
        init_gemini(Config.GEMINI_API_KEY)
        logger.info("LLM service initialized successfully")

    except Exception as e:
        logger.critical(f"Service initialization error: {e}", exc_info=True)
        raise

    # Register blueprints
    from stressease.api.mood import mood_bp
    from stressease.api.chat import chat_bp
    from stressease.api.predict import predict_bp
    from stressease.api.analytics import analytics_bp

    app.register_blueprint(mood_bp, url_prefix="/api/mood")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")

    # Global error handlers
    @app.errorhandler(400)
    def bad_request(error):
        logger.warning(f"Bad request: {error}")
        return (
            jsonify(
                {
                    "success": False,
                    "error_code": "BAD_REQUEST",
                    "message": "The request could not be understood by the server",
                }
            ),
            400,
        )

    @app.errorhandler(401)
    def unauthorized(error):
        logger.warning(f"Unauthorized access attempt: {error}")
        return (
            jsonify(
                {
                    "success": False,
                    "error_code": "AUTHENTICATION_ERROR",
                    "message": "Authentication required",
                }
            ),
            401,
        )

    @app.errorhandler(403)
    def forbidden(error):
        logger.warning(f"Forbidden access: {error}")
        return (
            jsonify(
                {
                    "success": False,
                    "error_code": "AUTHORIZATION_ERROR",
                    "message": "Access denied",
                }
            ),
            403,
        )

    @app.errorhandler(404)
    def not_found(error):
        return (
            jsonify(
                {
                    "success": False,
                    "error_code": "NOT_FOUND",
                    "message": "The requested resource was not found",
                }
            ),
            404,
        )

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}", exc_info=True)
        return (
            jsonify(
                {
                    "success": False,
                    "error_code": "SERVER_ERROR",
                    "message": "An unexpected error occurred",
                }
            ),
            500,
        )

    # Health check endpoint
    @app.route("/health")
    def health_check():
        """
        Health check endpoint that actually tests service availability.

        Returns:
            200 OK if all services are healthy
            503 Service Unavailable if any critical service is down
        """
        health_status = {
            "status": "healthy",
            "services": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        all_healthy = True

        # Test Firebase connection
        try:
            from stressease.services.utility.firebase_config import get_firestore_client

            db = get_firestore_client()
            # Simple check: ensure db client exists
            if db is None:
                raise RuntimeError("Firestore client is None")
            health_status["services"]["firebase"] = "healthy"
        except Exception as e:
            logger.error(f"Firebase health check failed: {e}")
            health_status["services"]["firebase"] = "unhealthy"
            all_healthy = False

        # Test Gemini API availability
        try:
            from stressease.services.chat.llm_service import (
                get_base_model,
                get_advance_model,
            )

            base_model = get_base_model()
            advance_model = get_advance_model()
            if base_model is None or advance_model is None:
                raise RuntimeError("LLM models not initialized")
            health_status["services"]["gemini_api"] = "healthy"
        except Exception as e:
            logger.error(f"Gemini API health check failed: {e}")
            health_status["services"]["gemini_api"] = "unhealthy"
            all_healthy = False

        # Test configuration
        try:
            if not Config.GEMINI_API_KEY or not Config.FIREBASE_CREDENTIALS_PATH:
                raise ValueError("Missing critical configuration")
            health_status["services"]["config"] = "healthy"
        except Exception as e:
            logger.error(f"Configuration health check failed: {e}")
            health_status["services"]["config"] = "unhealthy"
            all_healthy = False

        # Set overall status
        if not all_healthy:
            health_status["status"] = "unhealthy"
            return jsonify(health_status), 503

        return jsonify(health_status), 200

    # API root endpoint
    @app.route("/api")
    def api_root():
        return (
            jsonify(
                {
                    "message": "Welcome to StressEase Backend API",
                    "version": "1.0.0",
                    "endpoints": {
                        "mood": "/api/mood",
                        "chat": "/api/chat",
                        "predict": "/api/predict",
                        "analytics": "/api/analytics",
                    },
                }
            ),
            200,
        )

    return app
