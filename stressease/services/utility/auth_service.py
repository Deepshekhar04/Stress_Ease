"""Auth helpers and @token_required decorator."""

import functools
from flask import request, jsonify, g
import firebase_admin.auth
import logging

logger = logging.getLogger(__name__)


def token_required(f):
    """Verify Firebase ID token from Authorization: Bearer <token> and pass user_id to the route."""

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "AUTHENTICATION_ERROR",
                        "message": "Authorization header is required",
                    }
                ),
                401,
            )

        # Check if header follows "Bearer <token>" format
        try:
            scheme, token = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authorization scheme")
        except ValueError:
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "AUTHENTICATION_ERROR",
                        "message": "Authorization header must be in format: Bearer <token>",
                    }
                ),
                401,
            )

        # Validate the Firebase JWT token
        try:
            # Verify the ID token and decode it
            decoded_token = firebase_admin.auth.verify_id_token(token)
            user_id = decoded_token["uid"]

            # Store user_id in Flask's g object for access in other functions
            g.current_user_id = user_id

            # Pass user_id as the first argument to the decorated function
            return f(user_id, *args, **kwargs)

        except firebase_admin.auth.InvalidIdTokenError:
            logger.warning("Invalid Firebase ID token provided")
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "AUTHENTICATION_ERROR",
                        "message": "Invalid or expired token",
                    }
                ),
                401,
            )
        except firebase_admin.auth.ExpiredIdTokenError:
            logger.warning("Expired Firebase ID token provided")
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "AUTHENTICATION_ERROR",
                        "message": "Token has expired",
                    }
                ),
                401,
            )
        except firebase_admin.auth.RevokedIdTokenError:
            logger.warning("Revoked Firebase ID token provided")
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "AUTHENTICATION_ERROR",
                        "message": "Token has been revoked",
                    }
                ),
                401,
            )
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}", exc_info=True)
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "AUTHENTICATION_ERROR",
                        "message": "Token validation failed",
                    }
                ),
                401,
            )

    return decorated_function


# Note: Additional auth helper functions can be added here as needed
