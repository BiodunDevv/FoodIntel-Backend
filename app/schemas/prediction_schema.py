from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.food_schema import NutritionValues
from app.schemas.meal_schema import MealLogPublic


class PredictionResult(BaseModel):
    meal: MealLogPublic
    nutrition_available: bool
    notes: list[str] = Field(default_factory=list)


class PredictionUploadExample(BaseModel):
    serving_size_g: float | None = Field(default=250, examples=[250])


class PredictionImageUrlRequest(BaseModel):
    image_url: str = Field(
        ...,
        description="A publicly reachable food image URL, such as a Cloudinary unsigned upload URL.",
        examples=["https://res.cloudinary.com/demo/image/upload/sample.jpg"],
    )
    serving_size_g: float | None = Field(default=250, examples=[250])


class MLStatusPayload(BaseModel):
    model_loaded: bool
    model_version: str
    class_count: int

    model_config = ConfigDict(protected_namespaces=())


class PredictionFeedbackRequest(BaseModel):
    meal_id: str = Field(..., description="Meal log ID for the prediction being reviewed.")
    is_correct: bool = Field(..., description="Whether the prediction was correct.")
    corrected_slug: str | None = Field(
        default=None,
        description="Correct class slug if the prediction was wrong.",
        examples=["jollof_rice"],
    )
    notes: str | None = Field(
        default=None,
        description="Optional reviewer or user notes about the correction.",
        max_length=500,
    )


class PredictionFeedbackPayload(BaseModel):
    feedback_id: str
    meal_id: str
    status: str
    retraining_triggered: bool = False
    retraining_status: str | None = None
    retraining_message: str | None = None
