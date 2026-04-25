from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: T


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: str | None = None


class ModelBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class PyObjectId(str):
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        json_schema = handler(core_schema)
        json_schema.update(type="string")
        return json_schema


class TimestampedModel(ModelBase):
    created_at: datetime | None = None
    updated_at: datetime | None = None
