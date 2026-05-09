from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.database.mongodb import get_collection
from app.schemas.common import APIResponse
from app.schemas.report_schema import WeeklyReportPayload
from app.schemas.user_schema import UserPublic
from app.services.auth_service import get_current_user
from app.services.report_service import build_report_for_range


router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get(
    "/weekly",
    response_model=APIResponse[WeeklyReportPayload],
    summary="Generate nutrition report for a date range",
    description=(
        "Aggregate the authenticated user's meal logs for a given date range. "
        "Defaults to the current calendar month when no dates are provided. "
        "Pass start_date and end_date (YYYY-MM-DD) to query any custom range."
    ),
)
async def get_weekly_report(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[WeeklyReportPayload]:
    today = datetime.now(timezone.utc).date()

    if start_date:
        range_start = date.fromisoformat(start_date)
    else:
        range_start = today.replace(day=1)

    if end_date:
        range_end = date.fromisoformat(end_date)
    else:
        # last day of current month
        if today.month == 12:
            range_end = today.replace(day=31)
        else:
            range_end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)

    start_dt = datetime(range_start.year, range_start.month, range_start.day, tzinfo=timezone.utc)
    end_dt = datetime(range_end.year, range_end.month, range_end.day, 23, 59, 59, tzinfo=timezone.utc)

    cursor = get_collection("meal_logs").find(
        {"user_id": current_user.id, "created_at": {"$gte": start_dt, "$lte": end_dt}}
    )
    meals = [meal async for meal in cursor]
    report = build_report_for_range(meals, range_start, range_end)
    return APIResponse(message="Report generated successfully.", data=report)
