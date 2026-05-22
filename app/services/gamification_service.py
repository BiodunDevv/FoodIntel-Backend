from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone

from app.database.mongodb import get_collection
from app.schemas.user_schema import UserBadge, UserProgressPayload, UserQuest


BADGE_DEFINITIONS = (
    {
        "id": "first_scan",
        "label": "First Scan",
        "description": "Log your first meal with FoodIntel.",
        "target": 1,
        "metric": "total_scans",
    },
    {
        "id": "meal_explorer",
        "label": "Meal Explorer",
        "description": "Log 5 meals to build your history.",
        "target": 5,
        "metric": "total_scans",
    },
    {
        "id": "streak_starter",
        "label": "Streak Starter",
        "description": "Keep a 3-day meal logging streak alive.",
        "target": 3,
        "metric": "streak_days",
    },
    {
        "id": "nutrition_tracker",
        "label": "Nutrition Tracker",
        "description": "Log 5 meals with nutrition-backed food matches.",
        "target": 5,
        "metric": "nutrition_ready_meals",
    },
    {
        "id": "sharp_reviewer",
        "label": "Sharp Reviewer",
        "description": "Submit your first model correction.",
        "target": 1,
        "metric": "correction_count",
    },
    {
        "id": "feedback_coach",
        "label": "Feedback Coach",
        "description": "Submit feedback on 3 predictions.",
        "target": 3,
        "metric": "feedback_count",
    },
)


def _meal_day(value: datetime | None) -> date | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).date()


def _compute_streak(days: Iterable[date]) -> int:
    unique_days = sorted(set(days), reverse=True)
    if not unique_days:
        return 0

    today = datetime.now(timezone.utc).date()
    if unique_days[0] not in {today, today - timedelta(days=1)}:
        return 0

    streak = 1
    previous = unique_days[0]
    for day in unique_days[1:]:
        if previous - day == timedelta(days=1):
            streak += 1
            previous = day
        elif previous == day:
            continue
        else:
            break
    return streak


def _level_progress(xp_total: int) -> tuple[int, int, int]:
    level = max(1, (xp_total // 100) + 1)
    level_floor = (level - 1) * 100
    xp_into_level = xp_total - level_floor
    xp_for_next_level = 100
    return level, xp_into_level, xp_for_next_level


async def build_user_progress(user_id: str) -> UserProgressPayload:
    meal_logs = [
        meal
        async for meal in get_collection("meal_logs").find({"user_id": user_id})
    ]
    feedback_items = [
        feedback
        async for feedback in get_collection("prediction_feedback").find({"user_id": user_id})
    ]

    meal_days = [day for meal in meal_logs if (day := _meal_day(meal.get("created_at")))]
    streak_days = _compute_streak(meal_days)
    total_scans = len(meal_logs)
    nutrition_ready_meals = sum(1 for meal in meal_logs if meal.get("food_id"))
    high_confidence_scans = sum(
        1 for meal in meal_logs if float(meal.get("confidence") or 0) >= 0.85
    )

    feedback_count = len(feedback_items)
    correction_count = sum(1 for feedback in feedback_items if not feedback.get("is_correct"))
    confirmation_count = feedback_count - correction_count

    today = datetime.now(timezone.utc).date()
    meals_today = [
        meal for meal in meal_logs if _meal_day(meal.get("created_at")) == today
    ]
    feedback_today = [
        feedback for feedback in feedback_items if _meal_day(feedback.get("created_at")) == today
    ]

    xp_total = (
        total_scans * 20
        + nutrition_ready_meals * 5
        + confirmation_count * 10
        + correction_count * 25
        + min(streak_days, 14) * 3
        + high_confidence_scans * 2
    )
    level, xp_into_level, xp_for_next_level = _level_progress(xp_total)

    metrics = {
        "total_scans": total_scans,
        "streak_days": streak_days,
        "nutrition_ready_meals": nutrition_ready_meals,
        "feedback_count": feedback_count,
        "correction_count": correction_count,
    }

    badges = [
        UserBadge(
            id=badge["id"],
            label=badge["label"],
            description=badge["description"],
            unlocked=metrics[badge["metric"]] >= badge["target"],
            progress_current=min(metrics[badge["metric"]], badge["target"]),
            progress_target=badge["target"],
            unlocked_at=datetime.now(timezone.utc)
            if metrics[badge["metric"]] >= badge["target"]
            else None,
        )
        for badge in BADGE_DEFINITIONS
    ]

    quests = [
        UserQuest(
            id="scan_today",
            label="Scan a meal today",
            completed=bool(meals_today),
            xp=20,
        ),
        UserQuest(
            id="nutrition_today",
            label="Log one nutrition-backed meal",
            completed=any(meal.get("food_id") for meal in meals_today),
            xp=15,
        ),
        UserQuest(
            id="feedback_today",
            label="Review one prediction today",
            completed=bool(feedback_today),
            xp=20,
        ),
        UserQuest(
            id="keep_streak",
            label="Keep your streak alive",
            completed=streak_days > 0 and bool(meals_today),
            xp=25,
        ),
    ]

    return UserProgressPayload(
        level=level,
        xp_total=xp_total,
        xp_into_level=xp_into_level,
        xp_for_next_level=xp_for_next_level,
        streak_days=streak_days,
        total_scans=total_scans,
        nutrition_ready_meals=nutrition_ready_meals,
        feedback_count=feedback_count,
        correction_count=correction_count,
        high_confidence_scans=high_confidence_scans,
        badges=badges,
        quests=quests,
    )
