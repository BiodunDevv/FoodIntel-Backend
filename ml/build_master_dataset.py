import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single canonical FoodIntel ImageFolder dataset by combining existing "
            "ImageFolder datasets and optional feedback/raw folders."
        )
    )
    parser.add_argument("--imagefolder-sources", nargs="*", default=[])
    parser.add_argument("--raw-sources", nargs="*", default=[])
    parser.add_argument("--class-map", default="")
    parser.add_argument("--feedback-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="ml/dataset_master")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def load_class_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key.strip().lower(): value.strip() for key, value in payload.items()}


def hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def collect_imagefolder_sources(
    source_dirs: list[str],
    merged_images: dict[str, list[Path]],
    seen_hashes: dict[str, set[str]],
) -> None:
    for source in source_dirs:
        source_dir = Path(source)
        for split in ("train", "val", "test"):
            split_dir = source_dir / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                for image_path in sorted(class_dir.iterdir()):
                    if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    image_hash = hash_file(image_path)
                    if image_hash in seen_hashes[class_dir.name]:
                        continue
                    seen_hashes[class_dir.name].add(image_hash)
                    merged_images[class_dir.name].append(image_path)


def collect_raw_sources(
    source_dirs: list[str],
    class_map: dict[str, str],
    merged_images: dict[str, list[Path]],
    seen_hashes: dict[str, set[str]],
) -> None:
    for source in source_dirs:
        source_dir = Path(source)
        for raw_folder in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            target_class = class_map.get(raw_folder.name.strip().lower())
            if not target_class:
                continue
            for image_path in sorted(raw_folder.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                image_hash = hash_file(image_path)
                if image_hash in seen_hashes[target_class]:
                    continue
                seen_hashes[target_class].add(image_hash)
                merged_images[target_class].append(image_path)


def collect_feedback_dirs(
    feedback_dirs: list[str],
    merged_images: dict[str, list[Path]],
    seen_hashes: dict[str, set[str]],
) -> None:
    for source in feedback_dirs:
        source_dir = Path(source)
        if not source_dir.exists():
            continue
        for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                image_hash = hash_file(image_path)
                if image_hash in seen_hashes[class_dir.name]:
                    continue
                seen_hashes[class_dir.name].add(image_hash)
                merged_images[class_dir.name].append(image_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    class_map = load_class_map(args.class_map)
    merged_images: dict[str, list[Path]] = defaultdict(list)
    seen_hashes: dict[str, set[str]] = defaultdict(set)

    collect_imagefolder_sources(args.imagefolder_sources, merged_images, seen_hashes)
    collect_raw_sources(args.raw_sources, class_map, merged_images, seen_hashes)
    collect_feedback_dirs(args.feedback_dirs, merged_images, seen_hashes)

    written_counts = Counter()

    for class_name, images in sorted(merged_images.items()):
        random.shuffle(images)
        splits = split_paths(images, args.val_split, args.test_split)
        for split_name, split_images in splits.items():
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(split_images, start=1):
                destination = target_dir / f"{class_name}_{index:05d}{image_path.suffix.lower()}"
                shutil.copy2(image_path, destination)
                written_counts[class_name] += 1

    print(f"Built canonical dataset at {output_dir}")
    print("Class counts:")
    for class_name, count in sorted(written_counts.items()):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
