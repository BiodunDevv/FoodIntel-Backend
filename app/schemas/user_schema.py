from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


GoalType = Literal["weight_loss", "muscle_gain", "maintenance", "general_health"]


class UserPublic(BaseModel):
    id: str = Field(..., examples=["6629a62f0d24ed2cf9f4d001"])
    full_name: str
    email: EmailStr
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    goal: GoalType | None = None
    activity_level: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UserUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=120)
    age: int | None = Field(default=None, ge=1, le=120)
    height_cm: float | None = Field(default=None, ge=30, le=300)
    weight_kg: float | None = Field(default=None, ge=1, le=500)
    goal: GoalType | None = None
    activity_level: str | None = Field(default=None, max_length=100)
