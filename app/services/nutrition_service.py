from __future__ import annotations

from app.schemas.food_schema import NutritionValues


def calculate_serving_nutrition(food_document: dict, serving_size_g: float | None) -> tuple[float, NutritionValues]:
    serving = serving_size_g or food_document["default_serving_size_g"]
    multiplier = serving / 100
    base = food_document["nutrition_per_100g"]
    nutrition = NutritionValues(
        calories=round(base.get("calories", 0) * multiplier, 2),
        protein=round(base.get("protein", 0) * multiplier, 2),
        carbs=round(base.get("carbs", 0) * multiplier, 2),
        fat=round(base.get("fat", 0) * multiplier, 2),
        fiber=round(base.get("fiber", 0) * multiplier, 2),
        sugar=round(base.get("sugar", 0) * multiplier, 2),
        sodium=round(base.get("sodium", 0) * multiplier, 2),
        iron=round(base.get("iron", 0) * multiplier, 2) if base.get("iron") is not None else None,
        calcium=round(base.get("calcium", 0) * multiplier, 2)
        if base.get("calcium") is not None
        else None,
        vitamin_c=round(base.get("vitamin_c", 0) * multiplier, 2)
        if base.get("vitamin_c") is not None
        else None,
    )
    return serving, nutrition
