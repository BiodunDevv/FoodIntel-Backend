from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile, status

from app.config.settings import get_settings
from app.database.mongodb import get_collection
from app.models.meal_model import build_meal_document
from app.schemas.common import APIResponse, ErrorResponse
from app.schemas.food_schema import NutritionValues
from app.schemas.meal_schema import MealLogPublic
from app.schemas.prediction_schema import (
    PredictionFeedbackPayload,
    PredictionFeedbackRequest,
    PredictionImageUrlRequest,
    PredictionResult,
)
from app.schemas.user_schema import UserPublic
from app.services.auth_service import get_current_user
from app.services.health_score_service import calculate_health_score
from app.services.ml_service import ml_service
from app.services.nutrition_service import calculate_serving_nutrition
from app.services.recommendation_service import generate_recommendations
from app.services.retraining_service import retraining_service
from app.services.storage_service import storage_service
from app.utils.image import open_image, open_image_from_url, validate_upload_image
from app.utils.object_id import serialize_mongo_document


router = APIRouter(prefix="/predictions", tags=["Predictions"])


async def persist_prediction_result(
    *,
    current_user: UserPublic,
    image_url: str,
    prediction: dict,
    serving_size_g: float | None,
) -> APIResponse[PredictionResult]:
    food_document = await get_collection("foods").find_one({"slug": prediction["slug"]})
    notes: list[str] = []

    if food_document is None:
        serving = serving_size_g or 100
        nutrition = NutritionValues()
        health_score = 0
        recommendations = [
            "Nutrition data is unavailable for this predicted food right now. Seed or add the food record first."
        ]
        food_id = None
        notes.append(
            "The model predicted a food class, but no matching nutrition record exists in the foods collection."
        )
    else:
        serving, nutrition = calculate_serving_nutrition(food_document, serving_size_g)
        health_score, health_reasons = calculate_health_score(nutrition)
        recommendations = health_reasons + generate_recommendations(nutrition, health_score)
        food_id = str(food_document["_id"])

    meal_payload = build_meal_document(
        {
            "user_id": current_user.id,
            "food_id": food_id,
            "image_url": image_url,
            "predicted_food": prediction["label"].replace("_", " ").title(),
            "predicted_slug": prediction["slug"],
            "confidence": prediction["confidence"],
            "serving_size_g": serving,
            "nutrition": nutrition.model_dump(),
            "health_score": health_score,
            "recommendations": recommendations,
            "model_version": prediction["model_version"],
        }
    )
    result = await get_collection("meal_logs").insert_one(meal_payload)
    meal = await get_collection("meal_logs").find_one({"_id": result.inserted_id})
    assert meal is not None

    meal_public = MealLogPublic.model_validate(serialize_mongo_document(meal))
    return APIResponse(
        message="Prediction completed successfully.",
        data=PredictionResult(
            meal=meal_public,
            nutrition_available=food_document is not None,
            notes=notes,
        ),
    )


@router.post(
    "/image",
    response_model=APIResponse[PredictionResult],
    summary="Upload an image for food prediction",
    description="Accept a food image, run inference, estimate nutrition, compute health score, and save a meal log.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image."},
        401: {"model": ErrorResponse, "description": "Unauthorized."},
        503: {"model": ErrorResponse, "description": "ML model unavailable."},
    },
)
async def predict_food_from_image(
    file: UploadFile = File(..., description="Food image file in jpg, jpeg, png, or webp format."),
    serving_size_g: float | None = Form(default=None),
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[PredictionResult]:
    settings = get_settings()
    if not ml_service.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not trained or loaded yet. Train a model and restart the backend.",
        )

    extension = validate_upload_image(file, settings.max_upload_size_mb * 1024 * 1024)
    image_url = await storage_service.save_upload(file, extension)
    local_path = storage_service.get_local_path(image_url)
    prediction = ml_service.predict(open_image(local_path))
    return await persist_prediction_result(
        current_user=current_user,
        image_url=image_url,
        prediction=prediction,
        serving_size_g=serving_size_g,
    )


@router.post(
    "/image-url",
    response_model=APIResponse[PredictionResult],
    summary="Predict from a remote image URL",
    description="Fetch a public image URL server-side, run inference, estimate nutrition, and save a meal log.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image URL."},
        401: {"model": ErrorResponse, "description": "Unauthorized."},
        503: {"model": ErrorResponse, "description": "ML model unavailable."},
    },
)
async def predict_food_from_image_url(
    payload: PredictionImageUrlRequest = Body(...),
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[PredictionResult]:
    if not ml_service.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not trained or loaded yet. Train a model and restart the backend.",
        )

    prediction = ml_service.predict(await open_image_from_url(payload.image_url))
    return await persist_prediction_result(
        current_user=current_user,
        image_url=payload.image_url,
        prediction=prediction,
        serving_size_g=payload.serving_size_g,
    )


@router.post(
    "/feedback",
    response_model=APIResponse[PredictionFeedbackPayload],
    summary="Submit prediction feedback for retraining review",
    description=(
        "Store user feedback for a predicted meal so it can be reviewed later and, once "
        "validated, exported into a retraining dataset. This does not auto-train the model."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid feedback payload."},
        401: {"model": ErrorResponse, "description": "Unauthorized."},
        404: {"model": ErrorResponse, "description": "Meal not found."},
    },
)
async def submit_prediction_feedback(
    payload: PredictionFeedbackRequest = Body(...),
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[PredictionFeedbackPayload]:
    try:
        meal_object_id = ObjectId(payload.meal_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid meal ID supplied for feedback.",
        ) from exc

    meal = await get_collection("meal_logs").find_one(
        {"_id": meal_object_id, "user_id": current_user.id}
    )
    if meal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meal not found for the current user.",
        )

    corrected_slug = payload.corrected_slug.strip().lower() if payload.corrected_slug else None
    if not payload.is_correct and not corrected_slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide corrected_slug when marking a prediction as incorrect.",
        )

    feedback_status = "confirmed" if payload.is_correct else "auto_approved"
    created_at = datetime.now(timezone.utc)
    cleaned_notes = payload.notes.strip() if payload.notes else None
    corrected_food = (
        meal["predicted_food"]
        if payload.is_correct
        else corrected_slug.replace("_", " ").title()
    )
    feedback_document = {
        "user_id": current_user.id,
        "meal_id": payload.meal_id,
        "image_url": meal["image_url"],
        "predicted_slug": meal["predicted_slug"],
        "predicted_food": meal["predicted_food"],
        "confidence": meal["confidence"],
        "is_correct": payload.is_correct,
        "corrected_slug": corrected_slug or meal["predicted_slug"],
        "notes": cleaned_notes,
        "status": feedback_status,
        "created_at": created_at,
    }
    result = await get_collection("prediction_feedback").insert_one(feedback_document)
    await get_collection("meal_logs").update_one(
        {"_id": meal_object_id, "user_id": current_user.id},
        {
            "$set": {
                "feedback": {
                    "is_correct": payload.is_correct,
                    "status": feedback_status,
                    "corrected_slug": corrected_slug or meal["predicted_slug"],
                    "corrected_food": corrected_food,
                    "notes": cleaned_notes,
                    "submitted_at": created_at,
                }
            }
        },
    )
    retraining_decision = (
        retraining_service.request_retraining(
            await get_collection("prediction_feedback").count_documents(
                {"status": {"$in": ["auto_approved", "approved"]}}
            )
        )
        if not payload.is_correct
        else None
    )

    return APIResponse(
        message=(
            retraining_decision.message
            if retraining_decision
            else "Feedback stored successfully."
        ),
        data=PredictionFeedbackPayload(
            feedback_id=str(result.inserted_id),
            meal_id=payload.meal_id,
            status=feedback_status,
            retraining_triggered=retraining_decision.triggered if retraining_decision else False,
            retraining_status=retraining_decision.status if retraining_decision else None,
            retraining_message=retraining_decision.message if retraining_decision else None,
        ),
    )
