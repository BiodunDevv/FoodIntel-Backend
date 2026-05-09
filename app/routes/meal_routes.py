from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.database.mongodb import get_collection
from app.schemas.common import APIResponse, ErrorResponse
from app.schemas.meal_schema import MealListPayload, MealLogPublic
from app.schemas.user_schema import UserPublic
from app.services.auth_service import get_current_user
from app.utils.object_id import serialize_mongo_document, to_object_id


router = APIRouter(prefix="/meals", tags=["Meals"])


@router.get(
    "",
    response_model=APIResponse[MealListPayload],
    summary="List meal history",
    description=(
        "Return the authenticated user's meal history in reverse chronological order. "
        "Pass date (YYYY-MM-DD) to filter to a single day. "
        "Limit is capped at 500 when a date filter is provided."
    ),
)
async def list_meals(
    limit: int = Query(10, ge=1, le=500),
    skip: int = Query(0, ge=0),
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[MealListPayload]:
    query: dict = {"user_id": current_user.id}

    if date:
        try:
            day_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")
        day_end = day_start.replace(hour=23, minute=59, second=59)
        query["created_at"] = {"$gte": day_start, "$lte": day_end}
        # allow larger limit for single-day queries
        effective_limit = min(limit if limit > 10 else 500, 500)
    else:
        effective_limit = limit

    collection = get_collection("meal_logs")
    cursor = collection.find(query).sort("created_at", 1 if date else -1).skip(skip).limit(effective_limit)
    items = [MealLogPublic.model_validate(serialize_mongo_document(item)) async for item in cursor]
    total = await collection.count_documents(query)
    return APIResponse(
        message="Meals retrieved successfully.",
        data=MealListPayload(items=items, total=total, limit=effective_limit, skip=skip),
    )


@router.get(
    "/{meal_id}",
    response_model=APIResponse[MealLogPublic],
    summary="Get meal log by id",
    description="Return a single meal log belonging to the authenticated user.",
    responses={404: {"model": ErrorResponse, "description": "Meal not found."}},
)
async def get_meal(
    meal_id: str,
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[MealLogPublic]:
    meal = await get_collection("meal_logs").find_one(
        {"_id": to_object_id(meal_id), "user_id": current_user.id}
    )
    if meal is None:
        raise HTTPException(status_code=404, detail="Meal log not found.")
    return APIResponse(
        message="Meal retrieved successfully.",
        data=MealLogPublic.model_validate(serialize_mongo_document(meal)),
    )


@router.delete(
    "/{meal_id}",
    response_model=APIResponse[dict],
    summary="Delete meal log",
    description="Delete a meal log belonging to the authenticated user.",
    responses={404: {"model": ErrorResponse, "description": "Meal not found."}},
)
async def delete_meal(
    meal_id: str,
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[dict]:
    result = await get_collection("meal_logs").delete_one(
        {"_id": to_object_id(meal_id), "user_id": current_user.id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Meal log not found.")
    return APIResponse(message="Meal deleted successfully.", data={"deleted": True})
