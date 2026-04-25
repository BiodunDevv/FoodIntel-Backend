from datetime import datetime, timezone
from typing import Any


def build_food_document(payload: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    document = dict(payload)
    document["created_at"] = now
    document["updated_at"] = now
    return document
