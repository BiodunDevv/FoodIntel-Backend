from datetime import datetime, timedelta, timezone

from app.schemas.food_schema import NutritionValues
from app.services.health_score_service import calculate_health_score
from app.services.nutrition_service import calculate_serving_nutrition
from app.services.recommendation_service import generate_recommendations
from app.services.report_service import build_weekly_report


def test_calculate_serving_nutrition() -> None:
    serving, nutrition = calculate_serving_nutrition(
        {
            "default_serving_size_g": 250,
            "nutrition_per_100g": {
                "calories": 150,
                "protein": 3.5,
                "carbs": 28,
                "fat": 4,
                "fiber": 1.5,
                "sugar": 2,
                "sodium": 300,
            },
        },
        200,
    )
    assert serving == 200
    assert nutrition.calories == 300
    assert nutrition.protein == 7


def test_health_score_and_recommendations() -> None:
    nutrition = NutritionValues(calories=720, protein=8, carbs=75, fat=32, fiber=3, sodium=900)
    score, reasons = calculate_health_score(nutrition)
    recommendations = generate_recommendations(nutrition, score)
    assert score < 100
    assert reasons
    assert recommendations


def test_weekly_report_aggregation() -> None:
    now = datetime.now(timezone.utc)
    meals = [
        {
            "created_at": now - timedelta(days=1),
            "nutrition": {"calories": 300, "protein": 10, "carbs": 40, "fat": 8, "fiber": 4, "sugar": 2, "sodium": 150},
            "health_score": 70,
            "predicted_food": "Jollof Rice",
        },
        {
            "created_at": now - timedelta(days=2),
            "nutrition": {"calories": 450, "protein": 18, "carbs": 55, "fat": 12, "fiber": 6, "sugar": 3, "sodium": 200},
            "health_score": 82,
            "predicted_food": "Beans",
        },
    ]
    report = build_weekly_report(meals)
    assert report.total_meals == 2
    assert report.total_calories == 750
    assert report.most_frequent_food in {"Jollof Rice", "Beans"}
