import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ml.common import build_model, load_classes, resolve_device
else:
    from ml.common import build_model, load_classes, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local prediction with a trained FoodIntel model.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--classes-path", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    classes = load_classes(Path(args.classes_path))
    checkpoint = torch.load(args.model_path, map_location=device)
    model = build_model(checkpoint.get("model_name", "mobilenet_v3_small"), len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_values, top_indices = torch.topk(probabilities, k=min(2, len(classes)))

    confidence = float(top_values[0].item())
    runner_up_confidence = float(top_values[1].item()) if len(top_values) > 1 else 0.0
    margin = confidence - runner_up_confidence
    accepted = confidence >= args.confidence_threshold and margin >= args.margin_threshold
    payload = {
        "accepted": accepted,
        "label": classes[int(top_indices[0].item())],
        "confidence": round(confidence, 4),
        "runner_up_label": classes[int(top_indices[1].item())] if len(top_indices) > 1 else None,
        "runner_up_confidence": round(runner_up_confidence, 4),
        "margin": round(margin, 4),
        "rejection_reason": None
        if accepted
        else "unsupported_or_uncertain_image",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
