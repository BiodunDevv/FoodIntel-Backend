from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.food_schema import NutritionValues


class MealLogPublic(BaseModel):
    id: str
    user_id: str
    food_id: str | None = None
    image_url: str
    predicted_food: str
    predicted_slug: str
    confidence: float
    serving_size_g: float = Field(..., gt=0)
    nutrition: NutritionValues
    health_score: int = Field(..., ge=0, le=100)
    recommendations: list[str]
    model_version: str
    created_at: datetime

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "id": "6629a62f0d24ed2cf9f4d550",
                "user_id": "6629a62f0d24ed2cf9f4d001",
                "food_id": "6629a62f0d24ed2cf9f4d100",
                "image_url": "/uploads/sample.jpg",
                "predicted_food": "Jollof Rice",
                "predicted_slug": "jollof_rice",
                "confidence": 0.9432,
                "serving_size_g": 250,
                "nutrition": {
                    "calories": 375,
                    "protein": 8.75,
                    "carbs": 70,
                    "fat": 10,
                    "fiber": 3.75,
                    "sugar": 5,
                    "sodium": 750,
                },
                "health_score": 68,
                "recommendations": ["Fiber is low, so the meal may be less filling and balanced."],
                "model_version": "mobilenet_v3_small-best",
                "created_at": "2026-04-24T12:00:00Z",
            }
        },
    )


class MealListPayload(BaseModel):
    items: list[MealLogPublic]
    total: int
    limit: int
    skip: int
