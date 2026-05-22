from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId


def to_object_id(value: str) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise ValueError("Invalid ObjectId.")
    return ObjectId(value)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def serialize_mongo_document(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = {k: _serialize_value(v) for k, v in document.items()}
    if "_id" in result:
        result["id"] = result.pop("_id")
    return result
