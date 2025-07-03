"""Basic authentication implementation"""

from typing import Dict, Optional
import json
import os
from datetime import datetime
from middleware.security_middleware import security


class BasicAuth:
    """Basic authentication with user management"""

    def __init__(self):
        self.users_file = "auth/users.json"
        self.users = self._load_users()

    def _load_users(self) -> Dict:
        """Load users from file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, "r") as f:
                return json.load(f)
        return {}

    def _save_users(self):
        """Save users to file"""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(self.users_file, "w") as f:
            json.dump(self.users, f, indent=2)

    def create_user(
        self, username: str, password: str, role: str = "viewer"
    ) -> bool:
        """Create a new user"""
        if username in self.users:
            return False

        self.users[username] = {
            "password": security.hash_password(password),
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": None,
        }
        self._save_users()
        return True

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return a token"""
        user = self.users.get(username)
        if not user:
            return None

        if security.verify_password(user["password"], password):
            # Update last login
            user["last_login"] = datetime.utcnow().isoformat()
            self._save_users()

            # Generate token
            return security.generate_token(username, user["role"])

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
            self.users[username]["password_reset_required"] = True
            self._save_users()


# Initialize auth
auth = BasicAuth()

# Create default admin user if none exists
if not auth.users:
    import secrets
    import string

    secure_password = "".join(
        secrets.choice(
            string.ascii_letters + string.digits + string.punctuation
        )
        for _ in range(16)
    )
    auth.create_user("admin", secure_password, "admin")
    auth.enforce_password_reset("admin")
    print(
        f"⚠️  Default admin user created with a secure password: {secure_password}"
    )
    print("⚠️  You must change this password on first login!")

# (Removed redundant block)
