from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import certifi
from pymongo import MongoClient

from app.config.settings import get_settings
from app.utils.food_seed import load_food_seed, resolve_project_path


logger = logging.getLogger("foodintel.import_food_nutrition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import FoodIntel food nutrition records from a JSON seed file."
    )
    parser.add_argument(
        "--input",
        default="app/data/food_nutrition_seed.json",
        help="Path to the JSON seed file.",
    )
    parser.add_argument("--upsert-only", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for script output.",
    )
    return parser.parse_args()


def build_client() -> MongoClient:
    settings = get_settings()
    client_kwargs = {}
    uri_lower = settings.mongodb_uri.lower()
    if "mongodb.net" in uri_lower and "tlscafile=" not in uri_lower:
        client_kwargs["tlsCAFile"] = certifi.where()
    return MongoClient(settings.mongodb_uri, **client_kwargs)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    settings = get_settings()
    input_path = resolve_project_path(args.input, must_exist=True)
    payload = load_food_seed(input_path)

    logger.info("Starting food nutrition import")
    logger.info("Seed file: %s", input_path)
    logger.info("Mongo database: %s", settings.mongodb_db_name)
    client = build_client()
    now = datetime.now(timezone.utc)
    inserted = 0
    updated = 0

    try:
        foods = client[settings.mongodb_db_name]["foods"]
        for item in payload:
            existing = foods.find_one({"slug": item["slug"]})
            document = {**item, "updated_at": now}
            if existing:
                foods.update_one({"_id": existing["_id"]}, {"$set": document})
                updated += 1
            elif not args.upsert_only:
                foods.insert_one({**document, "created_at": now})
                inserted += 1
    finally:
        client.close()

    logger.info("Imported food nutrition seed data from %s", input_path)
    logger.info("Processed: %s", len(payload))
    logger.info("Inserted: %s", inserted)
    logger.info("Updated: %s", updated)


if __name__ == "__main__":
    main()
