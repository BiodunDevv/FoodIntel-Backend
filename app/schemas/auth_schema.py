from pydantic import BaseModel, EmailStr, Field

from app.schemas.user_schema import UserPublic


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120, examples=["Ada Obi"])
    email: EmailStr = Field(..., examples=["ada@example.com"])
    password: str = Field(..., min_length=8, max_length=128, examples=["StrongPass123"])

    model_config = {
        "json_schema_extra": {
            "example": {
                "full_name": "Ada Obi",
                "email": "ada@example.com",
                "password": "StrongPass123",
            }
        }
    }


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., examples=["ada@example.com"])
    password: str = Field(..., min_length=8, max_length=128, examples=["StrongPass123"])

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "ada@example.com",
                "password": "StrongPass123",
            }
        }
    }


class TokenData(BaseModel):
    sub: str
    email: EmailStr


class AuthPayload(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
