import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local FoodIntel dataset and training pipeline.")
    parser.add_argument("--skip-smoke-train", action="store_true")
    parser.add_argument("--skip-full-train", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-images", type=int, default=2)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    try:
        import certifi

        env.setdefault("SSL_CERT_FILE", certifi.where())
        env.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass
    subprocess.run(command, cwd=BACKEND_ROOT, check=True, env=env)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def cleanup_notebooks() -> None:
    notebooks_dir = BACKEND_ROOT / "ml" / "notebooks"
    if not notebooks_dir.exists():
        return
    for notebook in notebooks_dir.glob("*.ipynb"):
        print(f"Removing notebook: {notebook.relative_to(BACKEND_ROOT)}")
        notebook.unlink()
    try:
        notebooks_dir.rmdir()
    except OSError:
        pass


def cleanup_old_generated_sources() -> None:
    for relative in ["ml/raw_nigerian_foods_a", "ml/raw_nigerian_foods_b"]:
        path = BACKEND_ROOT / relative
        if path.exists():
            print(f"Removing old generated source folder: {relative}")
            remove_path(path)


def assert_no_duplicate_classes(dataset_root: Path) -> None:
    duplicate_names = {
        "banga",
        "bitterleaf",
        "edikakong",
        "jollof_rice",
        "okra_soup",
    }
    present = {
        path.name
        for split in ["train", "val", "test"]
        for path in (dataset_root / split).glob("*")
        if path.is_dir()
    }
    duplicates = sorted(present & duplicate_names)
    if duplicates:
        raise SystemExit(f"Duplicate/unmerged classes remain: {', '.join(duplicates)}")


def find_first_test_image(dataset_root: Path) -> Path | None:
    for path in sorted((dataset_root / "test").glob("*/*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            return path
    return None


def print_dataset_summary(dataset_root: Path) -> None:
    summary: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        for class_dir in sorted(split_dir.glob("*")):
            if not class_dir.is_dir():
                continue
            summary.setdefault(class_dir.name, {})[split] = len(
                [
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ]
            )
    print("\nDataset split summary:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    python = sys.executable
    unified_dir = BACKEND_ROOT / "ml" / "data" / "raw" / "unified"
    dataset_root = BACKEND_ROOT / "ml" / "dataset_master"
    model_path = BACKEND_ROOT / "ml" / "models" / "food_model_extensive.pt"
    classes_path = BACKEND_ROOT / "ml" / "classes.json"

    run(
        [
            python,
            "ml/data/download_datasets.py",
            "--raw-root",
            "ml/data/raw",
            "--output-dir",
            str(unified_dir.relative_to(BACKEND_ROOT)),
            "--manifest-path",
            "ml/data/raw/dataset_manifest.json",
            "--archive-search-root",
            str(WORKSPACE_ROOT),
            "--source",
            "legacy_nigerian_a=ml/raw_nigerian_foods_a",
            "--source",
            "legacy_nigerian_b=ml/raw_nigerian_foods_b",
            "--clear-output",
            "--clear-extracted",
        ]
    )

    run(
        [
            python,
            "ml/prepare_imagefolder.py",
            "--input-dir",
            str(unified_dir.relative_to(BACKEND_ROOT)),
            "--output-dir",
            str(dataset_root.relative_to(BACKEND_ROOT)),
            "--clear-output",
            "--min-images",
            str(args.min_images),
        ]
    )
    assert_no_duplicate_classes(dataset_root)
    print_dataset_summary(dataset_root)

    cleanup_notebooks()
    cleanup_old_generated_sources()

    if not args.skip_smoke_train:
        run(
            [
                python,
                "ml/train.py",
                "--dataset-source",
                "imagefolder",
                "--data-dir",
                str(dataset_root.relative_to(BACKEND_ROOT)),
                "--model",
                "mobilenet_v3_small",
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--image-size",
                "160",
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--freeze-backbone",
                "--freeze-epochs",
                "1",
                "--early-stop-patience",
                "0",
                "--max-samples-per-class",
                "5",
                "--output",
                "ml/models/food_model_smoke.pt",
                "--classes-output",
                "ml/classes_smoke.json",
                "--reports-dir",
                "ml/reports/smoke",
            ]
        )

    if not args.skip_full_train:
        run(
            [
                python,
                "ml/train.py",
                "--dataset-source",
                "imagefolder",
                "--data-dir",
                str(dataset_root.relative_to(BACKEND_ROOT)),
                "--model",
                "mobilenet_v3_small",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--image-size",
                str(args.image_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--freeze-backbone",
                "--freeze-epochs",
                "2",
                "--early-stop-patience",
                "3",
                "--output",
                str(model_path.relative_to(BACKEND_ROOT)),
                "--classes-output",
                str(classes_path.relative_to(BACKEND_ROOT)),
                "--reports-dir",
                "ml/reports/local",
            ]
        )

        run(
            [
                python,
                "ml/evaluate.py",
                "--data-dir",
                str(dataset_root.relative_to(BACKEND_ROOT)),
                "--model-path",
                str(model_path.relative_to(BACKEND_ROOT)),
                "--classes-path",
                str(classes_path.relative_to(BACKEND_ROOT)),
                "--reports-dir",
                "ml/reports/local_eval",
                "--device",
                args.device,
            ]
        )

        test_image = find_first_test_image(dataset_root)
        if test_image is not None:
            run(
                [
                    python,
                    "ml/predict_local.py",
                    "--image",
                    str(test_image.relative_to(BACKEND_ROOT)),
                    "--model-path",
                    str(model_path.relative_to(BACKEND_ROOT)),
                    "--classes-path",
                    str(classes_path.relative_to(BACKEND_ROOT)),
                    "--device",
                    args.device,
                ]
            )


if __name__ == "__main__":
    main()
