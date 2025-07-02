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
