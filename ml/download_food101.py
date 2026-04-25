import argparse
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import Food101
from torchvision.datasets.utils import extract_archive


FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
FOOD101_ARCHIVE = "food-101.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Food-101 into ImageFolder format.")
    parser.add_argument("--root", required=True, help="Directory where Food-101 will be downloaded.")
    parser.add_argument("--output-dir", required=True, help="Output ImageFolder dataset directory.")
    parser.add_argument("--classes", default="", help="Comma-separated Food-101 classes to include.")
    parser.add_argument("--max-images-per-class", type=int, default=300)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Delete any existing Food-101 archive before downloading again.",
    )
    return parser.parse_args()


def cleanup_food101_download(root: Path) -> None:
    archive = root / FOOD101_ARCHIVE
    extracted_dir = root / "food-101"
    if archive.exists():
        archive.unlink()
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)


def download_with_curl(root: Path) -> None:
    archive = root / FOOD101_ARCHIVE
    if archive.exists():
        print(f"Reusing existing archive: {archive}")
    else:
        print(f"Downloading {FOOD101_URL} to {archive} with curl resume support")

    subprocess.run(
        [
            "curl",
            "-L",
            "-C",
            "-",
            "--fail",
            "--output",
            str(archive),
            FOOD101_URL,
        ],
        check=True,
    )

    extracted_dir = root / "food-101"
    if not extracted_dir.exists():
        print(f"Extracting {archive}...")
        extract_archive(str(archive), str(root))


def load_food101(root: Path, split: str, force_redownload: bool = False) -> Food101:
    if force_redownload:
        cleanup_food101_download(root)

    try:
        download_with_curl(root)
        return Food101(root=str(root), split=split, download=False)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        if "File not found or corrupted." not in str(exc):
            if isinstance(exc, subprocess.CalledProcessError):
                raise RuntimeError(
                    "Food-101 download failed. Re-run the command and curl will resume from the partial archive."
                ) from exc
            raise
        cleanup_food101_download(root)
        print("Detected a partial or corrupt Food-101 download. Removing it and retrying once...")
        download_with_curl(root)
        return Food101(root=str(root), split=split, download=False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_dataset = load_food101(root, split="train", force_redownload=args.force_redownload)
    test_dataset = load_food101(root, split="test")

    selected_classes = {item.strip() for item in args.classes.split(",") if item.strip()}
    pool: dict[str, list[Path]] = defaultdict(list)

    for image_path, label in zip(train_dataset._image_files, train_dataset._labels, strict=False):
        class_name = train_dataset.classes[label]
        if selected_classes and class_name not in selected_classes:
            continue
        pool[class_name].append(Path(image_path))

    for image_path, label in zip(test_dataset._image_files, test_dataset._labels, strict=False):
        class_name = test_dataset.classes[label]
        if selected_classes and class_name not in selected_classes:
            continue
        pool[class_name].append(Path(image_path))

    for class_name, paths in pool.items():
        random.shuffle(paths)
        limited = paths[: args.max_images_per_class]
        total = len(limited)
        test_count = max(1, int(total * args.test_split))
        val_count = max(1, int(total * args.val_split))
        train_count = max(1, total - val_count - test_count)

        splits = {
            "train": limited[:train_count],
            "val": limited[train_count : train_count + val_count],
            "test": limited[train_count + val_count : train_count + val_count + test_count],
        }

        for split_name, images in splits.items():
            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image in images:
                shutil.copy2(image, class_dir / image.name)

    print(f"Prepared Food-101 subset in {output_dir}")


if __name__ == "__main__":
    main()
