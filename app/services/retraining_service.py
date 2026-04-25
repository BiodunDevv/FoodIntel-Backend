import base64
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote_to_bytes

import certifi
import httpx
from PIL import Image
from pymongo import MongoClient

from app.config.settings import BACKEND_DIR, ROOT_DIR, get_settings
from app.services.ml_service import ml_service


@dataclass
class RetrainingDecision:
    triggered: bool
    status: str
    message: str


class RetrainingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = "idle"
        self._message = "No retraining job has been started yet."
        self._last_started_at: str | None = None
        self._last_finished_at: str | None = None
        self._last_exit_code: int | None = None

    def status(self) -> dict[str, str | int | None | bool]:
        with self._lock:
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "status": self._status,
                "message": self._message,
                "last_started_at": self._last_started_at,
                "last_finished_at": self._last_finished_at,
                "last_exit_code": self._last_exit_code,
            }

    def request_retraining(self, approved_count: int) -> RetrainingDecision:
        settings = get_settings()
        if not settings.retrain_on_feedback:
            self._set_state("disabled", "Automatic retraining is disabled by configuration.")
            return RetrainingDecision(
                triggered=False,
                status="disabled",
                message="Automatic retraining is disabled.",
            )

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return RetrainingDecision(
                    triggered=False,
                    status="running",
                    message="A retraining job is already running in the background.",
                )

        if approved_count < settings.retrain_min_feedback_samples:
            message = (
                f"Feedback saved. Retraining will start after "
                f"{settings.retrain_min_feedback_samples} approved samples "
                f"(currently {approved_count})."
            )
            self._set_state("waiting", message)
            return RetrainingDecision(triggered=False, status="waiting", message=message)

        thread = threading.Thread(target=self._run_pipeline, name="foodintel-retraining", daemon=True)
        with self._lock:
            self._thread = thread
            self._status = "queued"
            self._message = f"Retraining queued with {approved_count} approved feedback samples."
            self._last_started_at = datetime.now(timezone.utc).isoformat()
            self._last_finished_at = None
            self._last_exit_code = None
        thread.start()
        return RetrainingDecision(
            triggered=True,
            status="queued",
            message="Model fine-tuning has been queued and will run in the background.",
        )

    def _set_state(self, status: str, message: str, exit_code: int | None = None) -> None:
        with self._lock:
            self._status = status
            self._message = message
            self._last_exit_code = exit_code
            if status in {"completed", "failed"}:
                self._last_finished_at = datetime.now(timezone.utc).isoformat()

    def _build_mongo_client(self) -> MongoClient:
        settings = get_settings()
        client_kwargs = {}
        uri_lower = settings.mongodb_uri.lower()
        if "mongodb.net" in uri_lower and "tlscafile=" not in uri_lower:
            client_kwargs["tlsCAFile"] = certifi.where()
        return MongoClient(settings.mongodb_uri, **client_kwargs)

    def _prepare_working_dataset(self) -> int:
        settings = get_settings()
        base_dataset_path = settings.retrain_base_dataset_path
        working_dataset_path = settings.retrain_working_dataset_path
        feedback_export_path = settings.retrain_feedback_export_path
        uploads_path = settings.upload_path

        if not base_dataset_path.exists():
            raise RuntimeError(f"Base retraining dataset not found at {base_dataset_path}")

        if not working_dataset_path.exists():
            print(f"[Retraining] Creating working dataset from {base_dataset_path}")
            shutil.copytree(base_dataset_path, working_dataset_path)

        if feedback_export_path.exists():
            shutil.rmtree(feedback_export_path)
        feedback_export_path.mkdir(parents=True, exist_ok=True)

        client = self._build_mongo_client()
        exported = 0
        used_feedback_ids: list[str] = []
        try:
            collection = client[settings.mongodb_db_name]["prediction_feedback"]
            for feedback in collection.find({"status": {"$in": ["auto_approved", "approved"]}}):
                corrected_slug = feedback.get("corrected_slug")
                image_url = feedback.get("image_url")
                if not corrected_slug or not image_url:
                    continue

                extension = self._infer_extension(image_url)
                export_path = feedback_export_path / corrected_slug / f"{feedback['_id']}.{extension}"
                export_path.parent.mkdir(parents=True, exist_ok=True)

                if not self._materialize_image(image_url, export_path, uploads_path):
                    continue

                train_target_dir = working_dataset_path / "train" / corrected_slug
                train_target_dir.mkdir(parents=True, exist_ok=True)
                destination = train_target_dir / export_path.name
                shutil.copy2(export_path, destination)
                exported += 1
                used_feedback_ids.append(str(feedback["_id"]))
        finally:
            client.close()

        if used_feedback_ids:
            client = self._build_mongo_client()
            try:
                collection = client[settings.mongodb_db_name]["prediction_feedback"]
                from bson import ObjectId

                collection.update_many(
                    {"_id": {"$in": [ObjectId(feedback_id) for feedback_id in used_feedback_ids]}},
                    {
                        "$set": {
                            "status": "used_for_training",
                            "trained_at": datetime.now(timezone.utc),
                        }
                    },
                )
            finally:
                client.close()

        return exported

    def _infer_extension(self, image_url: str) -> str:
        lowered = image_url.lower()
        if lowered.startswith("data:image/png"):
            return "png"
        if lowered.startswith("data:image/webp"):
            return "webp"
        if lowered.startswith("data:image/jpeg") or lowered.startswith("data:image/jpg"):
            return "jpg"
        if lowered.endswith(".png"):
            return "png"
        if lowered.endswith(".webp"):
            return "webp"
        return "jpg"

    def _materialize_image(self, image_url: str, destination: Path, uploads_path: Path) -> bool:
        try:
            if image_url.startswith("/uploads/"):
                source_path = uploads_path / image_url.removeprefix("/uploads/")
                if not source_path.exists():
                    return False
                shutil.copy2(source_path, destination)
                return True

            if image_url.startswith("data:image/"):
                _, payload = image_url.split(",", 1)
                image_bytes = (
                    base64.b64decode(payload, validate=False)
                    if ";base64" in image_url[:80]
                    else unquote_to_bytes(payload)
                )
                return self._save_bytes_as_image(image_bytes, destination)

            if image_url.startswith("http://") or image_url.startswith("https://"):
                with httpx.Client(
                    timeout=20.0,
                    follow_redirects=True,
                    verify=certifi.where(),
                    headers={"User-Agent": "FoodIntelRetraining/1.0"},
                ) as client:
                    response = client.get(image_url)
                    response.raise_for_status()
                    return self._save_bytes_as_image(response.content, destination)
        except Exception as exc:
            print(f"[Retraining] Skipped feedback image {image_url}: {exc}")
            return False

        return False

    def _save_bytes_as_image(self, image_bytes: bytes, destination: Path) -> bool:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.save(destination)
        return True

    def _build_training_command(self) -> list[str]:
        settings = get_settings()
        command = [
            sys.executable,
            str(BACKEND_DIR / "ml/train.py"),
            "--dataset-source",
            "imagefolder",
            "--data-dir",
            str(settings.retrain_working_dataset_path),
            "--model",
            "mobilenet_v3_small",
            "--epochs",
            str(settings.retrain_epochs),
            "--batch-size",
            str(settings.retrain_batch_size),
            "--lr",
            str(settings.retrain_learning_rate),
            "--output",
            str(settings.resolved_model_path),
            "--classes-output",
            str(settings.resolved_class_names_path),
            "--reports-dir",
            str(BACKEND_DIR / "ml/reports"),
            "--device",
            settings.retrain_device,
            "--image-size",
            str(settings.retrain_image_size),
            "--num-workers",
            str(settings.retrain_num_workers),
            "--early-stop-patience",
            str(settings.retrain_early_stop_patience),
            "--resume-from",
            str(settings.resolved_model_path),
            "--freeze-backbone",
            "--freeze-epochs",
            "1",
        ]
        if settings.retrain_no_pretrained:
            command.append("--no-pretrained")
        return command

    def _run_pipeline(self) -> None:
        self._set_state("running", "Preparing feedback samples for retraining.")
        try:
            exported = self._prepare_working_dataset()
            if exported == 0:
                self._set_state(
                    "waiting",
                    "No usable feedback images were available for retraining yet.",
                )
                return

            print(f"[Retraining] Exported {exported} feedback images into the working dataset.")
            command = self._build_training_command()
            print(f"[Retraining] Starting background fine-tuning: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(f"[Retraining] {line.rstrip()}")

            exit_code = process.wait()
            if exit_code != 0:
                self._set_state("failed", "Background retraining failed.", exit_code=exit_code)
                print(f"[Retraining] Training process exited with code {exit_code}")
                return

            ml_service.load()
            self._set_state(
                "completed",
                "Background fine-tuning finished and the model was reloaded successfully.",
                exit_code=0,
            )
            print("[Retraining] Model reloaded successfully.")
        except Exception as exc:
            self._set_state("failed", f"Background retraining failed: {exc}", exit_code=1)
            print(f"[Retraining] Background job failed: {exc}")


retraining_service = RetrainingService()
