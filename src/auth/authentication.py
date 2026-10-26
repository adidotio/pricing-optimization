"""Authentication and cryptographic security functions."""
import os
import hmac
import hashlib
import json
import base64
import time
from typing import Optional, Dict, Any
from src.utils.config import settings
from src.utils.logger import logger


def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a cryptographically secure random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    salt_hex = salt.hex()
    key_hex = key.hex()
    return f"{salt_hex}${key_hex}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored salt$key hash."""
    try:
        if "$" not in hashed_password:
            return False
        salt_hex, key_hex = hashed_password.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(key, expected_key)
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def create_access_token(data: Dict[str, Any], expires_delta_minutes: Optional[int] = None) -> str:
    """Creates an HMAC-SHA256 signed JWT-like authorization token."""
    to_encode = data.copy()
    expire_minutes = expires_delta_minutes or settings.access_token_expire_minutes
    expire_timestamp = int(time.time()) + (expire_minutes * 60)
    to_encode.update({"exp": expire_timestamp, "iat": int(time.time())})

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(to_encode).encode()).decode().rstrip("=")

    message = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(settings.secret_key.encode(), message, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and cryptographically validates the token signature and expiration."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(settings.secret_key.encode(), message, hashlib.sha256).digest()
        
        # Add padding back if necessary
        sig_padding = len(sig_b64) % 4
        sig_b64_padded = sig_b64 + ("=" * (4 - sig_padding) if sig_padding else "")
        provided_sig = base64.urlsafe_b64decode(sig_b64_padded)

        if not hmac.compare_digest(expected_sig, provided_sig):
            return None

        # Decode payload
        payload_padding = len(payload_b64) % 4
        payload_b64_padded = payload_b64 + ("=" * (4 - payload_padding) if payload_padding else "")
        payload_json = base64.urlsafe_b64decode(payload_b64_padded).decode()
        payload = json.loads(payload_json)

        # Check expiration
        if "exp" in payload and payload["exp"] < int(time.time()):
            return None

        return payload
    except Exception as e:
        logger.debug(f"Failed to decode access token: {e}")
        return None
