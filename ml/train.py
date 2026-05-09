import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFile
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ml.common import build_model, freeze_backbone, resolve_device, save_classes, trainable_parameters, unfreeze_backbone
else:
    from ml.common import build_model, freeze_backbone, resolve_device, save_classes, trainable_parameters, unfreeze_backbone


class Food101Subset(Dataset):
    def __init__(self, base_dataset: datasets.Food101, allowed_indices: list[int], transform=None):
        self.base_dataset = base_dataset
        self.allowed_indices = allowed_indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.allowed_indices)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[self.allowed_indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FoodIntel image classifier.")
    parser.add_argument("--dataset-source", choices=["imagefolder", "food101"], required=True)
    parser.add_argument("--data-dir", default="ml/dataset")
    parser.add_argument("--food101-root", default="ml/raw_data")
    parser.add_argument("--selected-classes", default="")
    parser.add_argument("--model", choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"], default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="ml/models/food_model.pt")
    parser.add_argument("--classes-output", default="ml/classes.json")
    parser.add_argument("--reports-dir", default="ml/reports")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Optional cap per class for quick local smoke tests.",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional checkpoint path to resume/fine-tune from.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained weights and use random initialization.",
    )
    return parser.parse_args()


def resolve_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested

    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count - 1))


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_food101_loaders(args: argparse.Namespace, train_transform, eval_transform):
    root = args.food101_root
    selected_classes = {item.strip() for item in args.selected_classes.split(",") if item.strip()}
    raw_train = datasets.Food101(root=root, split="train", download=True)
    raw_test = datasets.Food101(root=root, split="test", download=True)

    def filter_indices(dataset: datasets.Food101) -> list[int]:
        if not selected_classes:
            return list(range(len(dataset)))
        return [
            index
            for index in range(len(dataset))
            if dataset.classes[dataset._labels[index]] in selected_classes
        ]

    train_indices = filter_indices(raw_train)
    test_indices = filter_indices(raw_test)

    class_names = sorted(selected_classes) if selected_classes else list(raw_train.classes)
    label_map = {raw_train.class_to_idx[name]: class_names.index(name) for name in class_names}

    class RemappedFood101Subset(Food101Subset):
        def __getitem__(self, index: int):
            image, label = super().__getitem__(index)
            return image, label_map[label]

    train_dataset = RemappedFood101Subset(raw_train, train_indices, transform=train_transform)
    held_out_dataset = RemappedFood101Subset(raw_test, test_indices, transform=eval_transform)
    val_size = max(1, int(len(held_out_dataset) * 0.5))
    test_size = max(1, len(held_out_dataset) - val_size)
    if test_size == 0:
        test_size = 1
        val_size = len(held_out_dataset) - 1
    val_dataset, test_dataset = random_split(held_out_dataset, [val_size, test_size])
    return train_dataset, val_dataset, test_dataset, class_names


def limit_imagefolder_samples(dataset: datasets.ImageFolder, max_samples_per_class: int) -> Subset | datasets.ImageFolder:
    if max_samples_per_class <= 0:
        return dataset

    counts: dict[int, int] = {}
    selected_indices: list[int] = []
    for index, (_, class_index) in enumerate(dataset.samples):
        count = counts.get(class_index, 0)
        if count >= max_samples_per_class:
            continue
        counts[class_index] = count + 1
        selected_indices.append(index)
    return Subset(dataset, selected_indices)


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB")
        return True
    except Exception as exc:
        print(f"Skipping unreadable image: {path} ({exc})")
        return False


def filter_unreadable_images(dataset: datasets.ImageFolder) -> datasets.ImageFolder:
    valid_samples = [(path, target) for path, target in dataset.samples if is_valid_image(path)]
    skipped = len(dataset.samples) - len(valid_samples)
    if skipped:
        print(f"Skipped {skipped} unreadable image(s) from {dataset.root}.")
    dataset.samples = valid_samples
    dataset.imgs = valid_samples
    dataset.targets = [target for _, target in valid_samples]
    return dataset


def create_imagefolder_loaders(args: argparse.Namespace, train_transform, eval_transform):
    data_dir = Path(args.data_dir)
    train_dataset = filter_unreadable_images(
        datasets.ImageFolder(data_dir / "train", transform=train_transform)
    )
    val_dataset = filter_unreadable_images(
        datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    )
    test_dataset = filter_unreadable_images(
        datasets.ImageFolder(data_dir / "test", transform=eval_transform)
    )
    class_names = list(train_dataset.classes)
    train_dataset = limit_imagefolder_samples(train_dataset, args.max_samples_per_class)
    val_dataset = limit_imagefolder_samples(val_dataset, args.max_samples_per_class)
    test_dataset = limit_imagefolder_samples(test_dataset, args.max_samples_per_class)
    return train_dataset, val_dataset, test_dataset, class_names


def evaluate_model(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            predictions.extend(preds.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    return total_loss / max(1, total), correct / max(1, total), labels, predictions


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def render_progress(
    *,
    epoch: int,
    total_epochs: int,
    batch_index: int,
    total_batches: int,
    running_loss: float,
    seen_samples: int,
    dataset_size: int,
    start_time: float,
) -> None:
    avg_loss = running_loss / max(1, seen_samples)
    progress = batch_index / max(1, total_batches)
    bar_width = 28
    filled = min(bar_width, int(progress * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.time() - start_time
    eta = (elapsed / max(1, batch_index)) * max(0, total_batches - batch_index)
    message = (
        f"\rEpoch {epoch}/{total_epochs} "
        f"[{bar}] {batch_index:>4}/{total_batches:<4} "
        f"loss={avg_loss:.4f} "
        f"samples={seen_samples}/{dataset_size} "
        f"elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)}"
    )
    print(message, end="", flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    train_transform, eval_transform = build_transforms(args.image_size)
    num_workers = resolve_num_workers(args.num_workers)
    if device.type == "cpu" and args.batch_size < 24:
        print(
            "Tip: CPU training can be slow. Consider --batch-size 24 or 32 if memory allows, "
            "--image-size 160, and --freeze-backbone with pretrained weights."
        )

    if args.dataset_source == "food101":
        train_dataset, val_dataset, test_dataset, class_names = create_food101_loaders(args, train_transform, eval_transform)
    else:
        train_dataset, val_dataset, test_dataset, class_names = create_imagefolder_loaders(args, train_transform, eval_transform)

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {resume_path} for fine-tuning.")

    if args.freeze_backbone:
        freeze_backbone(model, args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr)

    best_accuracy = 0.0
    history = []
    epochs_without_improvement = 0
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone and epoch == args.freeze_epochs + 1:
          unfreeze_backbone(model)
          optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
          print(f"Unfroze backbone at epoch {epoch}.")

        model.train()
        running_loss = 0.0
        total = 0
        epoch_start = time.time()
        total_batches = len(train_loader)
        dataset_size = len(train_dataset)
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            render_progress(
                epoch=epoch,
                total_epochs=args.epochs,
                batch_index=batch_index,
                total_batches=total_batches,
                running_loss=running_loss,
                seen_samples=total,
                dataset_size=dataset_size,
                start_time=epoch_start,
            )

        train_loss = running_loss / max(1, total)
        print()
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_accuracy, 4),
            }
        )
        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": args.model,
                    "model_version": f"{args.model}-best",
                },
                output_path,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping after epoch {epoch} because validation accuracy has not improved "
                f"for {args.early_stop_patience} epochs."
            )
            break

    classes_output = Path(args.classes_output)
    save_classes(classes_output, class_names)

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_accuracy, labels, predictions = evaluate_model(model, test_loader, device)

    report = classification_report(labels, predictions, target_names=class_names, zero_division=0)
    matrix = confusion_matrix(labels, predictions)
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names).plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png")
    plt.close()

    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (reports_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (reports_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_accuracy": best_accuracy,
                "test_loss": round(test_loss, 4),
                "test_accuracy": round(test_accuracy, 4),
                "classes": class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training complete. Best val accuracy: {best_accuracy:.4f}, test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
