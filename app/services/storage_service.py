from pathlib import Path

import cloudinary
import cloudinary.uploader
from fastapi import UploadFile

from app.config.settings import get_settings
from app.utils.image import build_safe_filename


class StorageService:
    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.cloudinary_enabled:
            cloudinary.config(
                cloud_name=self.settings.cloudinary_cloud_name,
                api_key=self.settings.cloudinary_api_key,
                api_secret=self.settings.cloudinary_api_secret,
                secure=True,
            )

    async def save_upload(self, file: UploadFile, extension: str) -> str:
        upload_dir = self.settings.upload_path
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = build_safe_filename(extension)
        target = upload_dir / filename
        payload = await file.read()
        with target.open("wb") as handle:
            handle.write(payload)
        await file.seek(0)

        if self.settings.cloudinary_enabled:
            try:
                cloudinary.uploader.upload(str(target), folder="foodintel/uploads")
            except Exception:
                pass

        return f"/uploads/{filename}"

    def get_local_path(self, image_url: str) -> Path:
        relative = image_url.removeprefix("/uploads/")
        return self.settings.upload_path / relative


storage_service = StorageService()
