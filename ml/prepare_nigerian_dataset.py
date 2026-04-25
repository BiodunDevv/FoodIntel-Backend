import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map a raw Nigerian food dataset into FoodIntel ImageFolder classes."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Raw dataset directory with one folder per source class.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where train/val/test folders will be created.",
    )
    parser.add_argument(
        "--class-map",
        required=True,
        help="JSON file mapping raw folder names to FoodIntel class slugs.",
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the existing output directory before writing the split dataset.",
    )
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key.strip().lower(): value.strip() for key, value in payload.items()}


def list_images(folder: Path) -> list[Path]:
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def split_paths(paths: list[Path], val_split: float, test_split: float) -> dict[str, list[Path]]:
    total = len(paths)
    test_count = max(1, int(total * test_split)) if total >= 3 else max(0, int(total > 1))
    val_count = max(1, int(total * val_split)) if total >= 3 else max(0, int(total > 2))
    train_count = total - val_count - test_count

    if train_count <= 0:
        train_count = max(1, total - 2) if total >= 3 else max(1, total - 1)
        remaining = total - train_count
        val_count = 1 if remaining > 1 else 0
        test_count = remaining - val_count

    return {
        "train": paths[:train_count],
        "val": paths[train_count : train_count + val_count],
        "test": paths[train_count + val_count : train_count + val_count + test_count],
    }


def copy_split_files(splits: dict[str, list[Path]], output_dir: Path, target_class: str) -> None:
    for split_name, images in splits.items():
        class_dir = output_dir / split_name / target_class
        class_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            destination = class_dir / image.name
            if destination.exists():
                destination = class_dir / f"{image.stem}_{abs(hash(str(image))) % 10_000}{image.suffix}"
            shutil.copy2(image, destination)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    class_map = load_class_map(Path(args.class_map))

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    counts = Counter()
    skipped: dict[str, str] = {}
    unsupported = defaultdict(list)

    for raw_folder in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        raw_name = raw_folder.name.strip().lower()
        target_class = class_map.get(raw_name)
        if target_class is None:
            skipped[raw_folder.name] = "No mapping found"
            continue

        images = []
        for file_path in sorted(raw_folder.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                unsupported[raw_folder.name].append(file_path.name)
                continue
            images.append(file_path)

        if not images:
            skipped[raw_folder.name] = "No supported images found"
            continue

        random.shuffle(images)
        splits = split_paths(images, args.val_split, args.test_split)
        copy_split_files(splits, output_dir, target_class)
        counts[target_class] += len(images)

    print("Prepared Nigerian dataset for FoodIntel.")
    print("Class counts:")
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")

    if skipped:
        print("Skipped folders:")
        for folder, reason in sorted(skipped.items()):
            print(f"  {folder}: {reason}")

    if unsupported:
        print("Unsupported files:")
        for folder, files in sorted(unsupported.items()):
            print(f"  {folder}: {', '.join(files[:10])}")


if __name__ == "__main__":
    main()
