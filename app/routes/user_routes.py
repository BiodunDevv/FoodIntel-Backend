from fastapi import APIRouter, Depends

from app.schemas.common import APIResponse, ErrorResponse
from app.schemas.user_schema import UserPublic, UserUpdateRequest
from app.services.auth_service import get_current_user, update_current_user


router = APIRouter(prefix="/users", tags=["Users"])


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
