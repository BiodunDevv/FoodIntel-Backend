from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends

from app.database.mongodb import get_collection
from app.schemas.common import APIResponse
from app.schemas.report_schema import WeeklyReportPayload
from app.schemas.user_schema import UserPublic
from app.services.auth_service import get_current_user
from app.services.report_service import build_weekly_report


router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get(
    "/weekly",
    response_model=APIResponse[WeeklyReportPayload],
    summary="Generate weekly nutrition report",
    description="Aggregate the authenticated user's meal logs from the last 7 days.",
)
async def get_weekly_report(
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[WeeklyReportPayload]:
    start = datetime.now(timezone.utc) - timedelta(days=7)
    cursor = get_collection("meal_logs").find(
        {"user_id": current_user.id, "created_at": {"$gte": start}}
    )
    meals = [meal async for meal in cursor]
    report = build_weekly_report(meals)
    return APIResponse(message="Weekly report generated successfully.", data=report)
