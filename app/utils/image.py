import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote_to_bytes
from uuid import uuid4

import certifi
import httpx
from fastapi import HTTPException, UploadFile, status
from PIL import Image


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def validate_upload_image(file: UploadFile, max_size_bytes: int) -> str:
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Allowed types: jpg, jpeg, png, webp.",
        )

    size_header = file.headers.get("content-length")
    if size_header is not None and int(size_header) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded image exceeds the configured size limit.",
        )
    return suffix


def build_safe_filename(extension: str) -> str:
    return f"{uuid4().hex}.{extension}"


def open_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def open_image_from_bytes(image_bytes: bytes, invalid_detail: str) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=invalid_detail,
        ) from exc


def open_image_from_data_url(image_url: str) -> Image.Image:
    try:
        header, payload = image_url.split(",", 1)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided data URL is invalid.",
        ) from exc

    if not header.startswith("data:image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided data URL is not an image.",
        )

    if ";base64" in header:
        try:
            image_bytes = base64.b64decode(payload, validate=False)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The provided base64 image data could not be decoded.",
            ) from exc
    else:
        image_bytes = unquote_to_bytes(payload)

    return open_image_from_bytes(
        image_bytes,
        "The provided data URL could not be decoded as an image.",
    )


async def open_image_from_url(image_url: str, timeout: float = 20.0) -> Image.Image:
    if image_url.startswith("data:image/"):
        return open_image_from_data_url(image_url)

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            verify=certifi.where(),
            headers={"User-Agent": "FoodIntelBackend/1.0"},
        ) as client:
            response = await client.get(image_url)
            response.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to fetch the provided image URL.",
        ) from exc

    return open_image_from_bytes(
        response.content,
        "The provided image URL could not be decoded as an image.",
    )
