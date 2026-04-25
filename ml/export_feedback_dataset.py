import argparse
import os
import shutil
from pathlib import Path

from pymongo import MongoClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export reviewed prediction feedback into an ImageFolder-style dataset for the next "
            "training cycle. Feedback must be reviewed before export."
        )
    )
    parser.add_argument("--mongodb-uri", default=os.getenv("MONGODB_URI", ""))
    parser.add_argument("--db-name", default=os.getenv("MONGODB_DB_NAME", "foodintel_db"))
    parser.add_argument("--output-dir", default="ml/dataset_feedback")
    parser.add_argument("--uploads-dir", default="backend/uploads")
    parser.add_argument("--status", default="approved")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mongodb_uri:
        raise SystemExit("MONGODB_URI is required to export feedback samples.")

    output_dir = Path(args.output_dir)
    uploads_dir = Path(args.uploads_dir)

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    client = MongoClient(args.mongodb_uri)
    collection = client[args.db_name]["prediction_feedback"]

    exported = 0
    for feedback in collection.find({"status": args.status}):
        corrected_slug = feedback.get("corrected_slug")
        image_url = feedback.get("image_url")
        if not corrected_slug or not image_url:
            continue

        filename = image_url.removeprefix("/uploads/")
        source_path = uploads_dir / filename
        if not source_path.exists():
            continue

        target_dir = output_dir / corrected_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / source_path.name
        if destination.exists():
            destination = target_dir / f"{feedback['_id']}_{source_path.name}"

        shutil.copy2(source_path, destination)
        exported += 1

    print(f"Exported {exported} reviewed feedback images to {output_dir}")


if __name__ == "__main__":
    main()
