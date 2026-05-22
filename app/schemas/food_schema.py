from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class NutritionValues(BaseModel):
    calories: float = 0
    protein: float = 0
    carbs: float = 0
    fat: float = 0
    fiber: float = 0
    sugar: float = 0
    sodium: float = 0
    iron: float | None = None
    calcium: float | None = None
    vitamin_c: float | None = None


class FoodPublic(BaseModel):
    id: str
    name: str
    slug: str
    category: str
    description: str
    default_serving_size_g: float = Field(..., gt=0)
    nutrition_per_100g: NutritionValues
    is_local_food: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class FoodSeedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    slug: str
    category: str
    description: str
    default_serving_size_g: float = Field(..., gt=0)
    nutrition_per_100g: NutritionValues
    is_local_food: bool = True


class FoodsListPayload(BaseModel):
    items: list[FoodPublic]
    total: int


class SeedResult(BaseModel):
    inserted: int
    updated: int
