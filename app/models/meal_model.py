from datetime import datetime, timezone
from typing import Any


def build_meal_document(payload: dict[str, Any]) -> dict[str, Any]:
    document = dict(payload)
    document["created_at"] = datetime.now(timezone.utc)
    return document
