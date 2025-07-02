"""Enhanced security middleware for the application"""

import os
import jwt
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any

class SecurityMiddleware:
    """Security middleware with authentication and authorization"""
    
    def __init__(self):
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', '')
        self.api_secret = os.getenv('API_SECRET_KEY', '')
        
        if not self.jwt_secret or not self.api_secret:
            raise ValueError("Security keys not configured! Set JWT_SECRET_KEY and API_SECRET_KEY")
    
    def generate_token(self, user_id: str, role: str = 'viewer') -> str:
        """Generate a JWT token for a user"""
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def require_auth(self, required_role: Optional[str] = None):
        """Decorator to require authentication for a route"""
        def decorator(f):
            @wraps(f)
            async def decorated_function(*args, **kwargs):
                # Extract token from request headers
                token = self.extract_token_from_request()
                
                if not token:
                    return {'error': 'Authentication required'}, 401
                
                payload = self.verify_token(token)
                if not payload:
                    return {'error': 'Invalid or expired token'}, 401
                
                # Check role if required
                if required_role and payload.get('role') != required_role:
                    return {'error': 'Insufficient permissions'}, 403
                
                # Add user info to request context
                kwargs['user'] = payload
                return await f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def hash_password(self, password: str) -> str:
        """Hash a password using SHA256 with salt"""
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                      password.encode('utf-8'), 
                                      salt, 
                                      100000)
        return salt.hex() + pwdhash.hex()
    
    def verify_password(self, stored_password: str, provided_password: str) -> bool:
        """Verify a password against the stored hash"""
        salt = bytes.fromhex(stored_password[:64])
        stored_hash = stored_password[64:]
        pwdhash = hashlib.pbkdf2_hmac('sha256',
                                      provided_password.encode('utf-8'),
                                      salt,
                                      100000)
        return pwdhash.hex() == stored_hash
    
    def extract_token_from_request(self) -> Optional[str]:
        """Extract JWT token from request headers"""
        # This is a placeholder - implement based on your framework
        # For FastAPI: request.headers.get('Authorization', '').replace('Bearer ', '')
        # For Flask: request.headers.get('Authorization', '').replace('Bearer ', '')
        pass

# Initialize security middleware
security = SecurityMiddleware()
