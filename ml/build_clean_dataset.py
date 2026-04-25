import argparse
import shutil
from collections import Counter
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cleaner ImageFolder dataset by removing classes below a minimum sample "
            "threshold across train, val, and test."
        )
    )
    parser.add_argument("--input-dir", default="ml/dataset_extensive")
    parser.add_argument("--output-dir", default="ml/dataset_extensive_clean")
    parser.add_argument("--min-total-images", type=int, default=150)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    class_totals: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            class_totals[class_dir.name] += count_images(class_dir)

    kept_classes = {
        class_name for class_name, total in class_totals.items() if total >= args.min_total_images
    }

    if not kept_classes:
        raise SystemExit(
            f"No classes met the minimum threshold of {args.min_total_images} images."
        )

    written_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        split_dir = input_dir / split
        if not split_dir.exists():
            continue

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if class_dir.name not in kept_classes:
                continue

            destination_dir = output_dir / split / class_dir.name
            destination_dir.mkdir(parents=True, exist_ok=True)

            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                shutil.copy2(image_path, destination_dir / image_path.name)
                written_counts[class_dir.name] += 1

    print(f"Created clean dataset at {output_dir}")
    print(f"Minimum total images per class: {args.min_total_images}")
    print("Kept classes:")
    for class_name in sorted(kept_classes):
        print(f"  {class_name}: {class_totals[class_name]} images")

    dropped_classes = sorted(set(class_totals) - kept_classes)
    if dropped_classes:
        print("Dropped classes:")
        for class_name in dropped_classes:
            print(f"  {class_name}: {class_totals[class_name]} images")

    print("Written image counts:")
    for class_name, count in sorted(written_counts.items()):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
