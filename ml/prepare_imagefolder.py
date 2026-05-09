import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or split a custom food ImageFolder dataset.")
    parser.add_argument("--input-dir", required=True, help="Raw dataset directory with one folder per class.")
    parser.add_argument("--output-dir", help="Optional output directory for train/val/test split.")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true", help="Delete output-dir before writing splits.")
    parser.add_argument("--min-images", type=int, default=2, help="Skip classes with fewer supported images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    classes = [path for path in sorted(input_dir.iterdir()) if path.is_dir()]

    unsupported: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = {}

    for class_dir in classes:
        images = [path for path in class_dir.iterdir() if path.is_file()]
        supported_images = [path for path in images if path.suffix.lower() in SUPPORTED_EXTENSIONS]
        counts[class_dir.name] = len(supported_images)
        for image in images:
            if image.suffix.lower() not in SUPPORTED_EXTENSIONS:
                unsupported[class_dir.name].append(image.name)

    for name, count in counts.items():
        print(f"{name}: {count} images")
    if unsupported:
        print("Unsupported extensions found:")
        for name, files in unsupported.items():
            print(f"  {name}: {', '.join(files)}")

    if not args.output_dir:
        return

    output_dir = Path(args.output_dir)
    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    for class_dir in classes:
        images = [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(images) < args.min_images:
            print(f"Skipping {class_dir.name}: only {len(images)} supported image(s).")
            continue
        random.shuffle(images)
        total = len(images)
        test_count = int(total * args.test_split)
        val_count = int(total * args.val_split)
        if total >= 3:
            test_count = max(1, test_count)
            val_count = max(1, val_count)
        train_count = total - val_count - test_count
        if train_count <= 0:
            train_count = max(1, total - 1)
            val_count = max(0, total - train_count)
            test_count = 0
        splits = {
            "train": images[:train_count],
            "val": images[train_count : train_count + val_count],
            "test": images[train_count + val_count : train_count + val_count + test_count],
        }
        for split_name, selected in splits.items():
            target_dir = output_dir / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image in selected:
                shutil.copy2(image, target_dir / image.name)

    print(f"Prepared ImageFolder dataset at {output_dir}")


if __name__ == "__main__":
    main()
