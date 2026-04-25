from app.schemas.food_schema import NutritionValues


def generate_recommendations(nutrition: NutritionValues, health_score: int) -> list[str]:
    recommendations: list[str] = []

    if nutrition.protein < 10:
        recommendations.append("Your logged meal appears low in protein. Consider eggs, fish, chicken, beans, or moi moi.")
    if nutrition.carbs > 60 and nutrition.protein < 15:
        recommendations.append("The meal is carb-heavy. Reducing the portion or adding protein and vegetables may help balance it.")
    if nutrition.fiber < 5:
        recommendations.append("Fiber looks low. Consider adding vegetables, fruits, or beans.")
    if nutrition.sodium > 500:
        recommendations.append("Sodium is elevated. Try to reduce salty or highly processed sides where possible.")
    if health_score >= 80:
        recommendations.append("This looks fairly balanced overall. Keep building meals around this pattern.")

    if not recommendations:
        recommendations.append("This is an estimate, not medical advice. Keep tracking your meals for better weekly insights.")

    return recommendations
