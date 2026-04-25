from datetime import datetime, timezone
from typing import Any


def build_user_document(payload: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "full_name": payload["full_name"],
        "email": payload["email"].lower(),
        "password_hash": payload["password_hash"],
        "age": payload.get("age"),
        "height_cm": payload.get("height_cm"),
        "weight_kg": payload.get("weight_kg"),
        "goal": payload.get("goal"),
        "activity_level": payload.get("activity_level"),
        "created_at": now,
        "updated_at": now,
    }
