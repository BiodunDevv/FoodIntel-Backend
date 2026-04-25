from app.schemas.food_schema import NutritionValues


def calculate_health_score(nutrition: NutritionValues) -> tuple[int, list[str]]:
    score = 100
    reasons: list[str] = []

    if nutrition.calories > 700:
        score -= 20
        reasons.append("This serving is high in calories for one meal.")
    elif nutrition.calories > 500:
        score -= 10
        reasons.append("This serving is moderately high in calories.")

    if nutrition.protein < 10:
        score -= 15
        reasons.append("Protein is on the low side for satiety and recovery.")

    if nutrition.fiber < 5:
        score -= 15
        reasons.append("Fiber is low, so the meal may be less filling and balanced.")

    if nutrition.sodium > 800:
        score -= 20
        reasons.append("Sodium is high for a single serving.")
    elif nutrition.sodium > 500:
        score -= 10
        reasons.append("Sodium is moderately high.")

    if nutrition.fat > 30:
        score -= 10
        reasons.append("Fat content is high for one serving.")

    if nutrition.carbs > 60 and nutrition.protein < 15:
        score -= 10
        reasons.append("Carbohydrates are high relative to protein.")

    return max(0, min(100, score)), reasons
