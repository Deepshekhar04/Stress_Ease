"""Production WSGI entry point for cloud deployment (Render, Heroku, etc.)."""

from stressease import create_app
from config import Config

# Validate configuration before starting
try:
    Config.validate_config()
    print("[PASS] Configuration validation passed")
except ValueError as e:
    print(f"[ERROR] Configuration error: {e}")
    raise

# Create the Flask application instance for WSGI server
app = create_app()

if __name__ == "__main__":
    # This block is typically not executed in production
    # WSGI servers (gunicorn, etc.) import the 'app' object directly
    app.run()
