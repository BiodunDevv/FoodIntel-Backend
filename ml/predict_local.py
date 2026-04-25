import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ml.common import build_model, load_classes
else:
    from ml.common import build_model, load_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local prediction with a trained FoodIntel model.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--classes-path", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
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
        confidence, index = torch.max(probabilities, dim=0)
    print({"label": classes[index.item()], "confidence": round(float(confidence.item()), 4)})


if __name__ == "__main__":
    main()
