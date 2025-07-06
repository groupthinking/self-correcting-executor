#!/bin/bash
# Security Fix Script
# Addresses critical security issues from PROJECT_ANALYSIS_REPORT.md

echo "🔒 Starting security hardening..."

# Check if security_config.yaml exists
if [ ! -f "security_config.yaml" ]; then
    echo "❌ Error: security_config.yaml not found!"
    exit 1
fi

# Backup original config
echo "📦 Backing up original security config..."
cp security_config.yaml security_config.yaml.backup

# Create .env.example if it doesn't exist
echo "📄 Creating .env.example..."
cat << EOF > .env.example
# API Security
API_SECRET_KEY=your-secret-key-here-min-32-chars
JWT_SECRET_KEY=your-jwt-secret-here-min-32-chars

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379

# Encryption Keys
ENCRYPTION_KEY=your-256-bit-encryption-key-here

# External Services (if needed)
DWAVE_API_TOKEN=your-dwave-token-here
GCP_API_KEY=your-gcp-api-key-here

# Environment
ENVIRONMENT=development  # or production, staging
EOF

# Update security_config.yaml to enable security features
echo "🔧 Enabling security features..."

# Use Python to safely update YAML
python3 << EOF
import yaml
import sys

try:
    with open('security_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable authentication
    api_cfg = config.setdefault('api', {})
    auth_cfg = api_cfg.setdefault('authentication', {})
    auth_cfg['enabled'] = True
    print("✓ Enabled API authentication")
    
    # Enable sandboxing
    config['protocols']['sandboxing']['enabled'] = True
    print("✓ Enabled protocol sandboxing")
    
    # Enable encryption at rest
    config['data']['encryption']['at_rest'] = True
    print("✓ Enabled encryption at rest")
    
    # Add comments for TODOs
    with open('security_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✓ Updated security_config.yaml")
    
except Exception as e:
    print(f"❌ Error updating YAML: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to update security config. Restoring backup..."
    mv security_config.yaml.backup security_config.yaml
    exit 1
fi

# Create a secure random key generator
echo "🔑 Creating key generator script..."
cat << 'EOF' > scripts/generate-secure-keys.py
#!/usr/bin/env python3
"""Generate secure keys for the application"""

import secrets
import string
import base64

def generate_api_key(length=32):
    """Generate a secure API key"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_jwt_secret(length=64):
    """Generate a secure JWT secret"""
    return secrets.token_urlsafe(length)

def generate_encryption_key():
    """Generate a 256-bit encryption key"""
    return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')

if __name__ == "__main__":
    print("🔑 Secure Key Generator")
    print("=" * 50)
    print(f"API_SECRET_KEY={generate_api_key(32)}")
    print(f"JWT_SECRET_KEY={generate_jwt_secret()}")
    print(f"ENCRYPTION_KEY={generate_encryption_key()}")
    print("=" * 50)
    print("⚠️  Store these keys securely and never commit them to git!")
EOF

chmod +x scripts/generate-secure-keys.py

# Add security middleware
echo "🛡️ Creating security middleware..."
cat << 'EOF' > middleware/security_middleware.py
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
EOF

# Create a basic auth implementation
echo "🔐 Creating basic authentication module..."
mkdir -p auth
cat << 'EOF' > auth/basic_auth.py
"""Basic authentication implementation"""

from typing import Dict, Optional
import json
import os
from datetime import datetime
from middleware.security_middleware import security

class BasicAuth:
    """Basic authentication with user management"""
    
    def __init__(self):
        self.users_file = 'auth/users.json'
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load users from file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_users(self):
        """Save users to file"""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def create_user(self, username: str, password: str, role: str = 'viewer') -> bool:
        """Create a new user"""
        if username in self.users:
            return False
        
        self.users[username] = {
            'password': security.hash_password(password),
            'role': role,
            'created_at': datetime.utcnow().isoformat(),
            'last_login': None
        }
        self._save_users()
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return a token"""
        user = self.users.get(username)
        if not user:
            return None
        
        if security.verify_password(user['password'], password):
            # Update last login
            user['last_login'] = datetime.utcnow().isoformat()
            self._save_users()
            
            # Generate token
            return security.generate_token(username, user['role'])
        
        return None
    
    def delete_user(self, username: str) -> bool:
        """Delete a user"""
        if username in self.users:
            del self.users[username]
            self._save_users()
            return True
        return False

    def enforce_password_reset(self, username: str) -> None:
        """Enforce password reset for a user"""
        if username in self.users:
            self.users[username]['password_reset_required'] = True
            self._save_users()

# Initialize auth
auth = BasicAuth()

# Create default admin user if none exists
if not auth.users:
    import secrets
    import string
    secure_password = ''.join(secrets.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(16))
    auth.create_user('admin', secure_password, 'admin')
    auth.enforce_password_reset('admin')
    print(f"⚠️  Default admin user created with a secure password: {secure_password}")
    print("⚠️  You must change this password on first login!")

# Initialize auth
auth = BasicAuth()

# Create default admin user if none exists
if not auth.users:
    print("Creating default admin user...")
    import secrets
    import string
    secure_password = ''.join(secrets.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(16))
    auth.create_user('admin', secure_password, 'admin')
    print(f"⚠️  Default admin user created with a secure password: {secure_password}")
    print("⚠️  You must change this password on first login!")
EOF

# Update .gitignore to exclude sensitive files
echo "📄 Updating .gitignore for security files..."
cat << EOF >> .gitignore
# Security
.env
.env.*
!.env.example
auth/users.json
*.key
*.pem
*.cert
security_config.yaml.backup
EOF

# Summary
echo ""
echo "✅ Security Hardening Complete!"
echo "==============================="
echo ""
echo "🔒 Security features enabled:"
echo "   ✓ API Authentication"
echo "   ✓ Protocol Sandboxing"
echo "   ✓ Encryption at Rest"
echo ""
echo "📁 Files created:"
echo "   ✓ .env.example - Environment variable template"
echo "   ✓ scripts/generate-secure-keys.py - Secure key generator"
echo "   ✓ middleware/security_middleware.py - Security middleware"
echo "   ✓ auth/basic_auth.py - Basic authentication"
echo ""
echo "⚠️  IMPORTANT NEXT STEPS:"
echo "========================"
echo "1. Run: python3 scripts/generate-secure-keys.py"
echo "2. Copy the generated keys to a new .env file"
echo "3. Change the default admin password immediately"
echo "4. Review and adjust security settings in security_config.yaml"
echo "5. Test authentication before deploying"
echo ""
echo "🔑 To generate secure keys now, run:"
echo "   python3 scripts/generate-secure-keys.py"