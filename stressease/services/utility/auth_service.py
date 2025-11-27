"""Auth helpers and @token_required decorator."""

import functools
from flask import request, jsonify, g
import firebase_admin.auth

def token_required(f):
    """Verify Firebase ID token from Authorization: Bearer <token> and pass user_id to the route."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Extract token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authorization header is required'
            }), 401
        
        # Check if header follows "Bearer <token>" format
        try:
            scheme, token = auth_header.split(' ', 1)
            if scheme.lower() != 'bearer':
                raise ValueError("Invalid authorization scheme")
        except ValueError:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authorization header must be in format: Bearer <token>'
            }), 401
        
        # Validate the Firebase JWT token
        try:
            # Verify the ID token and decode it
            decoded_token = firebase_admin.auth.verify_id_token(token)
            user_id = decoded_token['uid']
            
            # Store user_id in Flask's g object for access in other functions
            g.current_user_id = user_id
            
            # Pass user_id as the first argument to the decorated function
            return f(user_id, *args, **kwargs)
            
        except firebase_admin.auth.InvalidIdTokenError:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid or expired token'
            }), 401
        except firebase_admin.auth.ExpiredIdTokenError:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Token has expired'
            }), 401
        except firebase_admin.auth.RevokedIdTokenError:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Token has been revoked'
            }), 401
        except Exception as e:
            # Log the error for debugging (in production, use proper logging)
            print(f"Token validation error: {str(e)}")
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Token validation failed'
            }), 401
    
    return decorated_function


# Note: Additional auth helper functions can be added here as needed