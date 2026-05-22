from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SETTINGS_FILE = Path(__file__).resolve()
STANDALONE_BACKEND_ROOT = SETTINGS_FILE.parents[2]
MONOREPO_ROOT_CANDIDATE = SETTINGS_FILE.parents[3]

if (MONOREPO_ROOT_CANDIDATE / "foodintel-backend" / "app").exists():
    ROOT_DIR = MONOREPO_ROOT_CANDIDATE
    BACKEND_DIR = ROOT_DIR / "foodintel-backend"
elif (MONOREPO_ROOT_CANDIDATE / "backend" / "app").exists():
    ROOT_DIR = MONOREPO_ROOT_CANDIDATE
    BACKEND_DIR = ROOT_DIR / "backend"
elif (STANDALONE_BACKEND_ROOT / "app").exists():
    ROOT_DIR = STANDALONE_BACKEND_ROOT
    BACKEND_DIR = ROOT_DIR
else:
    ROOT_DIR = STANDALONE_BACKEND_ROOT
    BACKEND_DIR = ROOT_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    mongodb_uri: str = Field(..., alias="MONGODB_URI")
    mongodb_db_name: str = Field("foodintel_db", alias="MONGODB_DB_NAME")
    jwt_secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(1440, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    upload_dir: str = Field("uploads", alias="UPLOAD_DIR")
    model_path: str = Field("ml/models/food_model_extensive.pt", alias="MODEL_PATH")
    class_names_path: str = Field("ml/classes.json", alias="CLASS_NAMES_PATH")
    environment: Literal["development", "test", "production"] = Field(
        "development", alias="ENVIRONMENT"
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="CORS_ORIGINS",
    )
    max_upload_size_mb: int = Field(10, alias="MAX_UPLOAD_SIZE_MB")
    cloudinary_cloud_name: str | None = Field(default=None, alias="CLOUDINARY_CLOUD_NAME")
    cloudinary_api_key: str | None = Field(default=None, alias="CLOUDINARY_API_KEY")
    cloudinary_api_secret: str | None = Field(default=None, alias="CLOUDINARY_API_SECRET")
    prediction_confidence_threshold: float = Field(0.55, alias="PREDICTION_CONFIDENCE_THRESHOLD")
    prediction_margin_threshold: float = Field(0.12, alias="PREDICTION_MARGIN_THRESHOLD")
    retrain_on_feedback: bool = Field(True, alias="RETRAIN_ON_FEEDBACK")
    retrain_min_feedback_samples: int = Field(5, alias="RETRAIN_MIN_FEEDBACK_SAMPLES")
    retrain_base_dataset_dir: str = Field("ml/dataset_master", alias="RETRAIN_BASE_DATASET_DIR")
    retrain_working_dataset_dir: str = Field("ml/dataset_live", alias="RETRAIN_WORKING_DATASET_DIR")
    retrain_feedback_export_dir: str = Field(
        "ml/dataset_feedback_live", alias="RETRAIN_FEEDBACK_EXPORT_DIR"
    )
    retrain_epochs: int = Field(2, alias="RETRAIN_EPOCHS")
    retrain_batch_size: int = Field(24, alias="RETRAIN_BATCH_SIZE")
    retrain_learning_rate: float = Field(0.00005, alias="RETRAIN_LEARNING_RATE")
    retrain_image_size: int = Field(160, alias="RETRAIN_IMAGE_SIZE")
    retrain_num_workers: int = Field(2, alias="RETRAIN_NUM_WORKERS")
    retrain_early_stop_patience: int = Field(1, alias="RETRAIN_EARLY_STOP_PATIENCE")
    retrain_no_pretrained: bool = Field(False, alias="RETRAIN_NO_PRETRAINED")
    retrain_device: Literal["cpu", "cuda", "auto"] = Field("cpu", alias="RETRAIN_DEVICE")
    admin_secret: str = Field("changeme-admin-2024", alias="ADMIN_SECRET")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @property
    def upload_path(self) -> Path:
        return self.resolve_runtime_path(self.upload_dir)

    @property
    def resolved_model_path(self) -> Path:
        return self.resolve_runtime_path(self.model_path)

    @property
    def resolved_class_names_path(self) -> Path:
        return self.resolve_runtime_path(self.class_names_path)

    @property
    def cloudinary_enabled(self) -> bool:
        return all(
            [
                self.cloudinary_cloud_name,
                self.cloudinary_api_key,
                self.cloudinary_api_secret,
            ]
        )

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def retrain_base_dataset_path(self) -> Path:
        return self.resolve_runtime_path(self.retrain_base_dataset_dir)

    @property
    def retrain_working_dataset_path(self) -> Path:
        return self.resolve_runtime_path(self.retrain_working_dataset_dir)

    @property
    def retrain_feedback_export_path(self) -> Path:
        return self.resolve_runtime_path(self.retrain_feedback_export_dir)

    def resolve_runtime_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path

        root_candidate = (ROOT_DIR / path).resolve()
        backend_candidate = (BACKEND_DIR / path).resolve()

        if root_candidate.exists():
            return root_candidate
        if backend_candidate.exists():
            return backend_candidate

        return root_candidate


@lru_cache
def get_settings() -> Settings:
    return Settings()
