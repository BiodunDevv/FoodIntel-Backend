from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


GoalType = Literal["weight_loss", "muscle_gain", "maintenance", "general_health"]


class UserBadge(BaseModel):
    id: str
    label: str
    description: str
    unlocked: bool
    progress_current: int
    progress_target: int
    unlocked_at: datetime | None = None


class UserQuest(BaseModel):
    id: str
    label: str
    completed: bool
    xp: int


class UserProgressPayload(BaseModel):
    level: int
    xp_total: int
    xp_into_level: int
    xp_for_next_level: int
    streak_days: int
    total_scans: int
    nutrition_ready_meals: int
    feedback_count: int
    correction_count: int
    high_confidence_scans: int
    badges: list[UserBadge]
    quests: list[UserQuest]


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

    @field_validator("goal", mode="before")
    @classmethod
    def normalize_goal_aliases(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        return {
            "lose_weight": "weight_loss",
            "gain_muscle": "muscle_gain",
            "maintain": "maintenance",
            "eat_healthier": "general_health",
        }.get(value, value)
