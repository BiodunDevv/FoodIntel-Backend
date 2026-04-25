from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.config.settings import get_settings
from app.database.mongodb import get_collection
from app.schemas.common import APIResponse, ErrorResponse
from app.schemas.food_schema import FoodPublic, FoodsListPayload, SeedResult
from app.utils.food_seed import load_food_seed
from app.utils.object_id import serialize_mongo_document


router = APIRouter(prefix="/foods", tags=["Foods"])


@router.get(
    "",
    response_model=APIResponse[FoodsListPayload],
    summary="List supported foods",
    description="Return all supported foods with placeholder nutrition values per 100g.",
)
async def list_foods() -> APIResponse[FoodsListPayload]:
    foods_collection = get_collection("foods")
    items = [FoodPublic.model_validate(serialize_mongo_document(item)) async for item in foods_collection.find().sort("name", 1)]
    return APIResponse(message="Foods retrieved successfully.", data=FoodsListPayload(items=items, total=len(items)))


@router.get(
    "/{slug}",
    response_model=APIResponse[FoodPublic],
    summary="Get food by slug",
    description="Return a supported food item by slug.",
    responses={404: {"model": ErrorResponse, "description": "Food not found."}},
)
async def get_food(slug: str) -> APIResponse[FoodPublic]:
    food = await get_collection("foods").find_one({"slug": slug})
    if food is None:
        raise HTTPException(status_code=404, detail="Food not found.")
    return APIResponse(
        message="Food retrieved successfully.",
        data=FoodPublic.model_validate(serialize_mongo_document(food)),
    )


@router.post(
    "/seed",
    response_model=APIResponse[SeedResult],
    summary="Seed default foods",
    description="Insert or update the development nutrition seed dataset for FoodIntel.",
    responses={403: {"model": ErrorResponse, "description": "Only available in development."}},
)
async def seed_foods() -> APIResponse[SeedResult]:
    settings = get_settings()
    if not settings.is_development:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This route is only enabled in development.")

    seed_foods_payload = load_food_seed()
    foods = get_collection("foods")
    inserted = 0
    updated = 0
    now = datetime.now(timezone.utc)

    for item in seed_foods_payload:
        existing = await foods.find_one({"slug": item["slug"]})
        payload = {**item, "updated_at": now}
        if existing:
            await foods.update_one({"_id": existing["_id"]}, {"$set": payload})
            updated += 1
        else:
            await foods.insert_one({**payload, "created_at": now})
            inserted += 1

    return APIResponse(
        message="Food seed completed successfully.",
        data=SeedResult(inserted=inserted, updated=updated),
    )
