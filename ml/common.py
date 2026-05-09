import json
from typing import Iterable
from pathlib import Path

import torch
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    def resolve_weights():
        if not pretrained:
            return None
        if model_name == "mobilenet_v3_small":
            return models.MobileNet_V3_Small_Weights.DEFAULT
        if model_name == "mobilenet_v3_large":
            return models.MobileNet_V3_Large_Weights.DEFAULT
        if model_name == "efficientnet_b0":
            return models.EfficientNet_B0_Weights.DEFAULT
        if model_name == "resnet18":
            return models.ResNet18_Weights.DEFAULT
        return None

    weights = resolve_weights()

    try:
        if model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if model_name == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=weights)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if model_name == "resnet18":
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Warning: failed to load pretrained weights ({exc}). Falling back to randomly initialized weights.")
        return build_model(model_name, num_classes, pretrained=False)

    raise ValueError(f"Unsupported model architecture: {model_name}")


def freeze_backbone(model: torch.nn.Module, model_name: str) -> None:
    if model_name.startswith("mobilenet_v3"):
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        return

    if model_name == "efficientnet_b0":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        return

    if model_name == "resnet18":
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False
        return

    raise ValueError(f"Unsupported model architecture for freezing: {model_name}")


def unfreeze_backbone(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def resolve_device(choice: str = "auto") -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_classes(path: Path, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")


def load_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["classes"] if isinstance(payload, dict) else list(payload)
