"""Flask app factory. Registers blueprints and initializes services."""

from flask import Flask, jsonify
from config import Config


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

    # Initialize services
    from stressease.services.firebase_config import init_firebase
    from stressease.services.llm_service import init_gemini

    try:
        # Initialize Firebase
        init_firebase(Config.FIREBASE_CREDENTIALS_PATH)
        print("✓ Firebase initialized successfully")

        # Initialize Gemini AI (dual-model LLM)
        init_gemini(Config.GEMINI_API_KEY)
        print("✓ LLM service initialized successfully")

    except Exception as e:
        print(f"✗ Service initialization error: {e}")
        raise

    # Register blueprints
    from stressease.api.mood import mood_bp
    from stressease.api.chat import chat_bp

    app.register_blueprint(mood_bp, url_prefix="/api/mood")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")

    # Global error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return (
            jsonify(
                {
                    "error": "Bad Request",
                    "message": "The request could not be understood by the server",
                }
            ),
            400,
        )

    @app.errorhandler(401)
    def unauthorized(error):
        return (
            jsonify({"error": "Unauthorized", "message": "Authentication required"}),
            401,
        )

    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({"error": "Forbidden", "message": "Access denied"}), 403

    @app.errorhandler(404)
    def not_found(error):
        return (
            jsonify(
                {
                    "error": "Not Found",
                    "message": "The requested resource was not found",
                }
            ),
            404,
        )

    @app.errorhandler(500)
    def internal_error(error):
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                }
            ),
            500,
        )

    # Health check endpoint
    @app.route("/health")
    def health_check():
        return (
            jsonify(
                {"status": "healthy", "message": "StressEase Backend API is running"}
            ),
            200,
        )

    # API root endpoint
    @app.route("/api")
    def api_root():
        return (
            jsonify(
                {
                    "message": "Welcome to StressEase Backend API",
                    "version": "1.0.0",
                    "endpoints": {"mood": "/api/mood", "chat": "/api/chat"},
                }
            ),
            200,
        )

    return app
