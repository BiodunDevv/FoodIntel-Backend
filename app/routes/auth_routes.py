from fastapi import APIRouter, Depends, status

from app.schemas.auth_schema import AuthPayload, LoginRequest, RegisterRequest
from app.schemas.common import APIResponse, ErrorResponse
from app.services.auth_service import get_current_user, login_user, register_user
from app.schemas.user_schema import UserPublic


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=APIResponse[AuthPayload],
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a FoodIntel user account, hash the password, and return a JWT access token.",
    responses={409: {"model": ErrorResponse, "description": "Email already exists."}},
)
async def register(payload: RegisterRequest) -> APIResponse[AuthPayload]:
    token, user = await register_user(payload)
    return APIResponse(
        message="User registered successfully.",
        data=AuthPayload(access_token=token, user=user),
    )


@router.post(
    "/login",
    response_model=APIResponse[AuthPayload],
    summary="Login user",
    description="Authenticate a user with email and password, then return a JWT access token.",
    responses={401: {"model": ErrorResponse, "description": "Invalid credentials."}},
)
async def login(payload: LoginRequest) -> APIResponse[AuthPayload]:
    token, user = await login_user(payload)
    return APIResponse(
        message="Login successful.",
        data=AuthPayload(access_token=token, user=user),
    )


@router.get(
    "/me",
    response_model=APIResponse[UserPublic],
    summary="Get current user",
    description="Return the currently authenticated user from the Bearer token.",
    responses={401: {"model": ErrorResponse, "description": "Unauthorized."}},
)
async def me(current_user: UserPublic = Depends(get_current_user)) -> APIResponse[UserPublic]:
    return APIResponse(message="Current user retrieved successfully.", data=current_user)
