from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from app.config.settings import get_settings


class LowConfidencePredictionError(RuntimeError):
    def __init__(
        self,
        *,
        label: str,
        confidence: float,
        runner_up_label: str | None,
        runner_up_confidence: float,
        margin: float,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.runner_up_label = runner_up_label
        self.runner_up_confidence = runner_up_confidence
        self.margin = margin
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        super().__init__(
            "The image does not look confidently like a supported food class."
        )


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    from torchvision import models

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model architecture: {model_name}")


class MLService:
    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = []
        self.model_version = "unavailable"
        self.model_loaded = False
        self.device = torch.device("cpu")
        self.error: str | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> None:
        settings = get_settings()
        model_path = settings.resolved_model_path
        classes_path = settings.resolved_class_names_path
        self.model_loaded = False
        self.error = None

        if not model_path.exists() or not classes_path.exists():
            self.model = None
            self.class_names = []
            self.model_version = "missing"
            return

        try:
            classes_payload = json.loads(classes_path.read_text(encoding="utf-8"))
            self.class_names = (
                classes_payload["classes"]
                if isinstance(classes_payload, dict)
                else list(classes_payload)
            )
            checkpoint = torch.load(model_path, map_location="cpu")
            architecture = checkpoint.get("model_name", "mobilenet_v3_small")
            model = build_model(architecture, len(self.class_names))
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.model_version = checkpoint.get("model_version", architecture)
            self.model_loaded = True
        except Exception as exc:  # pragma: no cover - runtime safety
            self.model = None
            self.class_names = []
            self.model_version = "error"
            self.error = str(exc)

    def status(self) -> dict[str, Any]:
        settings = get_settings()
        return {
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "class_count": len(self.class_names),
            "classes": self.class_names,
            "confidence_threshold": settings.prediction_confidence_threshold,
            "margin_threshold": settings.prediction_margin_threshold,
            "error": self.error,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("The ML model is not trained or loaded yet.")

        settings = get_settings()
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_values, top_indices = torch.topk(probabilities, k=min(2, len(self.class_names)))

        confidence = float(top_values[0].item())
        index = int(top_indices[0].item())
        label = self.class_names[index]
        runner_up_confidence = float(top_values[1].item()) if len(top_values) > 1 else 0.0
        runner_up_label = (
            self.class_names[int(top_indices[1].item())] if len(top_indices) > 1 else None
        )
        margin = confidence - runner_up_confidence

        if (
            confidence < settings.prediction_confidence_threshold
            or margin < settings.prediction_margin_threshold
        ):
            raise LowConfidencePredictionError(
                label=label,
                confidence=round(confidence, 4),
                runner_up_label=runner_up_label,
                runner_up_confidence=round(runner_up_confidence, 4),
                margin=round(margin, 4),
                confidence_threshold=settings.prediction_confidence_threshold,
                margin_threshold=settings.prediction_margin_threshold,
            )

        return {
            "label": label,
            "slug": label,
            "confidence": round(confidence, 4),
            "runner_up_label": runner_up_label,
            "runner_up_confidence": round(runner_up_confidence, 4),
            "margin": round(margin, 4),
            "model_version": self.model_version,
        }


ml_service = MLService()
