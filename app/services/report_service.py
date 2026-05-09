from collections import Counter, defaultdict
from datetime import date, timedelta

from app.schemas.report_schema import DailyCalorieTrendEntry, MacroTotals, WeeklyReportPayload


def build_report_for_range(meals: list[dict], start: date, end: date) -> WeeklyReportPayload:
    grouped = defaultdict(lambda: {"calories": 0.0, "meals": 0})
    macros = {
        "calories": 0.0,
        "protein": 0.0,
        "carbs": 0.0,
        "fat": 0.0,
        "fiber": 0.0,
        "sugar": 0.0,
        "sodium": 0.0,
    }
    score_total = 0
    food_counter: Counter[str] = Counter()

    for meal in meals:
        created_at = meal["created_at"]
        day_key = created_at.date().isoformat()
        grouped[day_key]["calories"] += meal["nutrition"]["calories"]
        grouped[day_key]["meals"] += 1
        for key in macros:
            macros[key] += meal["nutrition"].get(key, 0) or 0
        score_total += meal.get("health_score", 0)
        food_counter[meal.get("predicted_food", "Unknown")] += 1

    num_days = (end - start).days + 1
    trend = []
    for offset in range(num_days):
        current = (start + timedelta(days=offset)).isoformat()
        trend.append(
            DailyCalorieTrendEntry(
                date=current,
                calories=round(grouped[current]["calories"], 2),
                meals=grouped[current]["meals"],
            )
        )

    total_meals = len(meals)
    total_calories = round(macros["calories"], 2)
    recommendations: list[str] = []
    average_score = round(score_total / total_meals, 2) if total_meals else 0.0

    if total_meals == 0:
        recommendations.append("No meals were logged in this period yet.")
    elif average_score < 60:
        recommendations.append("Your meals trend lower on balance. Try increasing vegetables, fiber, and lean protein.")
    elif average_score >= 80:
        recommendations.append("Your pattern looks fairly balanced overall. Keep the consistency going.")

    return WeeklyReportPayload(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        total_meals=total_meals,
        total_calories=total_calories,
        average_calories_per_day=round(total_calories / num_days, 2),
        macro_totals=MacroTotals(**{key: round(value, 2) for key, value in macros.items()}),
        average_health_score=average_score,
        most_frequent_food=food_counter.most_common(1)[0][0] if food_counter else None,
        daily_calorie_trend=trend,
        recommendations=recommendations
        or ["This report is an estimate and not medical advice. Use it as a reflection tool, not a diagnosis."],
    )


# keep old name working for any other callers
def build_weekly_report(meals: list[dict]) -> WeeklyReportPayload:
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=6)
    return build_report_for_range(meals, start, today)
