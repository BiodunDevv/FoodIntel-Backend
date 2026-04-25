from pydantic import BaseModel


class MacroTotals(BaseModel):
    calories: float = 0
    protein: float = 0
    carbs: float = 0
    fat: float = 0
    fiber: float = 0
    sugar: float = 0
    sodium: float = 0


class DailyCalorieTrendEntry(BaseModel):
    date: str
    calories: float
    meals: int


class WeeklyReportPayload(BaseModel):
    start_date: str
    end_date: str
    total_meals: int
    total_calories: float
    average_calories_per_day: float
    macro_totals: MacroTotals
    average_health_score: float
    most_frequent_food: str | None
    daily_calorie_trend: list[DailyCalorieTrendEntry]
    recommendations: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "start_date": "2026-04-18",
                "end_date": "2026-04-24",
                "total_meals": 5,
                "total_calories": 2140,
                "average_calories_per_day": 305.71,
                "macro_totals": {
                    "calories": 2140,
                    "protein": 82,
                    "carbs": 255,
                    "fat": 67,
                    "fiber": 31,
                    "sugar": 22,
                    "sodium": 1840,
                },
                "average_health_score": 74.4,
                "most_frequent_food": "Beans",
                "daily_calorie_trend": [
                    {"date": "2026-04-18", "calories": 320, "meals": 1},
                    {"date": "2026-04-19", "calories": 0, "meals": 0},
                ],
                "recommendations": [
                    "Your weekly pattern looks fairly balanced overall. Keep the consistency going."
                ],
            }
        }
    }
