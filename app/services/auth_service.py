from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.database.mongodb import get_collection
from app.models.user_model import build_user_document
from app.schemas.auth_schema import LoginRequest, RegisterRequest
from app.schemas.user_schema import UserPublic, UserUpdateRequest
from app.utils.jwt import create_access_token, decode_access_token
from app.utils.object_id import serialize_mongo_document, to_object_id
from app.utils.password import hash_password, verify_password


security = HTTPBearer(auto_error=False)


def to_user_public(document: dict) -> UserPublic:
    safe_document = serialize_mongo_document(document) or {}
    safe_document.pop("password_hash", None)
    return UserPublic.model_validate(safe_document)


async def register_user(payload: RegisterRequest) -> tuple[str, UserPublic]:
    users = get_collection("users")
    existing = await users.find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists.")

    document = build_user_document(
        {
            "full_name": payload.full_name,
            "email": payload.email.lower(),
            "password_hash": hash_password(payload.password),
        }
    )
    result = await users.insert_one(document)
    user = await users.find_one({"_id": result.inserted_id})
    assert user is not None
    user_public = to_user_public(user)
    token = create_access_token(user_public.id, user_public.email)
    return token, user_public


async def login_user(payload: LoginRequest) -> tuple[str, UserPublic]:
    users = get_collection("users")
    user = await users.find_one({"email": payload.email.lower()})
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    user_public = to_user_public(user)
    token = create_access_token(user_public.id, user_public.email)
    return token, user_public


async def update_current_user(user_id: str, payload: UserUpdateRequest) -> UserPublic:
    users = get_collection("users")
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        user = await users.find_one({"_id": to_object_id(user_id)})
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        return to_user_public(user)

    updates["updated_at"] = datetime.now(timezone.utc)
    await users.update_one({"_id": to_object_id(user_id)}, {"$set": updates})
    user = await users.find_one({"_id": to_object_id(user_id)})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return to_user_public(user)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserPublic:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
        )

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload["sub"]
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        ) from exc

    users = get_collection("users")
    try:
        object_id = to_object_id(user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        ) from exc

    user = await users.find_one({"_id": object_id})
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return to_user_public(user)
