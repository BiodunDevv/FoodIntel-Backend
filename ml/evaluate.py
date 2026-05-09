import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ml.common import build_model, load_classes, resolve_device
else:
    from ml.common import build_model, load_classes, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FoodIntel model.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--classes-path", required=True)
    parser.add_argument("--reports-dir", default="ml/reports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
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

    test_dataset = datasets.ImageFolder(Path(args.data_dir) / "test", transform=transform)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
            labels.extend(targets.tolist())

    report = classification_report(labels, predictions, target_names=classes, zero_division=0)
    matrix = confusion_matrix(labels, predictions).tolist()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "evaluation_report.txt").write_text(report, encoding="utf-8")
    (reports_dir / "evaluation_confusion_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
