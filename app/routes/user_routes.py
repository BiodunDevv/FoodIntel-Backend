from fastapi import APIRouter, Depends

from app.schemas.common import APIResponse, ErrorResponse
from app.schemas.user_schema import UserProgressPayload, UserPublic, UserUpdateRequest
from app.services.auth_service import get_current_user, update_current_user
from app.services.gamification_service import build_user_progress


router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    response_model=APIResponse[UserPublic],
    summary="Get current user profile",
    description="Return the authenticated user's profile.",
    responses={401: {"model": ErrorResponse, "description": "Unauthorized."}},
)
async def get_me(
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[UserPublic]:
    return APIResponse(message="Profile retrieved successfully.", data=current_user)


@router.get(
    "/me/progress",
    response_model=APIResponse[UserProgressPayload],
    summary="Get current user progress and badges",
    description="Return the authenticated user's gamification progress, streaks, quests, and badges.",
    responses={401: {"model": ErrorResponse, "description": "Unauthorized."}},
)
async def get_my_progress(
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[UserProgressPayload]:
    progress = await build_user_progress(current_user.id)
    return APIResponse(message="Progress retrieved successfully.", data=progress)


@router.patch(
    "/me",
    response_model=APIResponse[UserPublic],
    summary="Update current user profile",
    description="Update the authenticated user's personal profile fields.",
    responses={401: {"model": ErrorResponse, "description": "Unauthorized."}},
)
async def update_me(
    payload: UserUpdateRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> APIResponse[UserPublic]:
    updated = await update_current_user(current_user.id, payload)
    return APIResponse(message="Profile updated successfully.", data=updated)
