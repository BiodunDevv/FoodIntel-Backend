from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config.settings import get_settings
from app.database.mongodb import get_collection
from app.services.retraining_service import retraining_service
from app.utils.object_id import serialize_mongo_document


router = APIRouter(tags=["Admin"])


def _require_admin(secret: str | None) -> None:
    settings = get_settings()
    if not secret or secret != settings.admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin secret.")


@router.get(
    "/admin/feedback",
    summary="List pending feedback items",
    include_in_schema=False,
)
async def list_feedback(
    secret: str | None = Query(None),
) -> JSONResponse:
    _require_admin(secret)

    collection = get_collection("prediction_feedback")
    pending_cursor = collection.find({"status": "auto_approved"}).sort("created_at", -1).limit(100)
    pending = [serialize_mongo_document(doc) async for doc in pending_cursor]

    approved_count = await collection.count_documents({"status": "approved"})
    used_count = await collection.count_documents({"status": "used_for_training"})
    rejected_count = await collection.count_documents({"status": "rejected"})

    settings = get_settings()

    return JSONResponse({
        "success": True,
        "data": {
            "pending_items": pending,
            "pending_count": len(pending),
            "approved_count": approved_count,
            "used_count": used_count,
            "rejected_count": rejected_count,
            "retraining_status": retraining_service.status(),
            "settings": {
                "retrain_min_feedback_samples": settings.retrain_min_feedback_samples,
                "prediction_confidence_threshold": settings.prediction_confidence_threshold,
                "prediction_margin_threshold": settings.prediction_margin_threshold,
            },
        },
    })


@router.post(
    "/admin/feedback/{feedback_id}/approve",
    summary="Approve a feedback item",
    include_in_schema=False,
)
async def approve_feedback(
    feedback_id: str,
    secret: str | None = Query(None),
) -> JSONResponse:
    _require_admin(secret)
    try:
        oid = ObjectId(feedback_id)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid feedback ID.")

    collection = get_collection("prediction_feedback")
    result = await collection.update_one(
        {"_id": oid},
        {"$set": {"status": "approved", "reviewed_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Feedback not found.")
    return JSONResponse({"success": True, "message": "Feedback approved."})


@router.post(
    "/admin/feedback/{feedback_id}/reject",
    summary="Reject a feedback item",
    include_in_schema=False,
)
async def reject_feedback(
    feedback_id: str,
    secret: str | None = Query(None),
) -> JSONResponse:
    _require_admin(secret)
    try:
        oid = ObjectId(feedback_id)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid feedback ID.")

    collection = get_collection("prediction_feedback")
    result = await collection.update_one(
        {"_id": oid},
        {"$set": {"status": "rejected", "reviewed_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Feedback not found.")
    return JSONResponse({"success": True, "message": "Feedback rejected."})


@router.post(
    "/admin/retrain",
    summary="Trigger immediate retraining",
    include_in_schema=False,
)
async def trigger_retrain(
    secret: str | None = Query(None),
    force: bool = Query(False, description="Bypass the minimum sample threshold"),
) -> JSONResponse:
    _require_admin(secret)

    collection = get_collection("prediction_feedback")
    approved_count = await collection.count_documents({"status": "approved"})

    if force:
        settings = get_settings()
        effective_count = max(approved_count, settings.retrain_min_feedback_samples)
    else:
        effective_count = approved_count

    decision = retraining_service.request_retraining(effective_count)
    return JSONResponse({
        "success": True,
        "triggered": decision.triggered,
        "status": decision.status,
        "message": decision.message,
        "approved_count": approved_count,
    })


@router.get(
    "/admin/retraining-status",
    summary="Get retraining job status",
    include_in_schema=False,
)
async def get_retraining_status(
    secret: str | None = Query(None),
) -> JSONResponse:
    _require_admin(secret)
    return JSONResponse({"success": True, "data": retraining_service.status()})


@router.get(
    "/admin/retraining-history",
    summary="Get retraining run history with logs",
    include_in_schema=False,
)
async def get_retraining_history(
    secret: str | None = Query(None),
) -> JSONResponse:
    _require_admin(secret)
    return JSONResponse({"success": True, "data": retraining_service.history()})
