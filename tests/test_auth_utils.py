from backend.app.utils.jwt import create_access_token, decode_access_token
from backend.app.utils.password import hash_password, verify_password


def test_password_hashing_roundtrip() -> None:
    password = "StrongPass123"
    password_hash = hash_password(password)
    assert password_hash != password
    assert verify_password(password, password_hash) is True


def test_access_token_roundtrip() -> None:
    token = create_access_token("user-1", "user@example.com")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-1"
    assert payload["email"] == "user@example.com"
