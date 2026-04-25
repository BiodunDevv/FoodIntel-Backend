import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas.food_schema import FoodSeedItem
from app.utils.food_seed import load_food_seed, load_slug_aliases, resolve_project_path


DEFAULT_CATEGORY = "Local Food"
DEFAULT_DESCRIPTION = "Imported from an external food composition table."
DEFAULT_SERVING_SIZE = 200


NAME_CANDIDATES = [
    "food_name",
    "name",
    "food",
    "food description",
    "description",
    "food item",
]
CALORIES_CANDIDATES = ["calories", "energy_kcal", "kcal", "energy (kcal)"]
PROTEIN_CANDIDATES = ["protein", "protein_g", "protein (g)"]
CARBS_CANDIDATES = ["carbs", "carbohydrate", "carbohydrate_g", "carbohydrate (g)"]
FAT_CANDIDATES = ["fat", "fat_g", "fat (g)", "lipid (g)"]
FIBER_CANDIDATES = ["fiber", "fibre", "fiber_g", "fibre (g)"]
SUGAR_CANDIDATES = ["sugar", "sugars", "sugar_g", "sugars (g)"]
SODIUM_CANDIDATES = ["sodium", "sodium_mg", "sodium (mg)"]
IRON_CANDIDATES = ["iron", "iron_mg", "iron (mg)"]
CALCIUM_CANDIDATES = ["calcium", "calcium_mg", "calcium (mg)"]
VITC_CANDIDATES = ["vitamin_c", "vitamin c", "vitamin c (mg)"]
SERVING_CANDIDATES = ["default_serving_size_g", "serving_size_g", "serving size (g)"]


logger = logging.getLogger("foodintel.convert_food_nutrition_table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a CSV/XLSX food composition table into FoodIntel's "
            "food_nutrition_seed.json format."
        )
    )
    parser.add_argument("--input", required=True, help="Path to CSV/XLSX nutrition table.")
    parser.add_argument(
        "--sheet",
        default="",
        help="Optional Excel sheet name. Ignored for CSV inputs.",
    )
    parser.add_argument(
        "--seed-input",
        default="backend/app/data/food_nutrition_seed.json",
        help="Existing FoodIntel seed file used as the base.",
    )
    parser.add_argument(
        "--aliases",
        default="backend/app/data/food_slug_aliases.json",
        help="Slug-to-alias mapping JSON file.",
    )
    parser.add_argument(
        "--output",
        default="backend/app/data/food_nutrition_seed.generated.json",
        help="Where to write the converted JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for script output.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize(value: str) -> str:
    lowered = value.strip().lower().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def slugify(value: str) -> str:
    return normalize(value).replace(" ", "_")


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_to_original = {normalize(column): column for column in columns}
    for candidate in candidates:
        found = normalized_to_original.get(normalize(candidate))
        if found:
            return found
    return None


def as_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        stripped = stripped.replace(",", "")
        try:
            return float(stripped)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_alias_lookup(aliases_path: Path) -> dict[str, str]:
    payload = load_slug_aliases(aliases_path)
    lookup: dict[str, str] = {}
    for slug, aliases in payload.items():
        lookup[normalize(slug)] = slug
        for alias in aliases:
            lookup[normalize(alias)] = slug
    return lookup


def load_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    input_path = resolve_project_path(args.input, must_exist=True)
    seed_input_path = resolve_project_path(args.seed_input, must_exist=True)
    aliases_path = resolve_project_path(args.aliases, must_exist=True)
    output_path = resolve_project_path(args.output, must_exist=False)

    logger.info("Starting nutrition table conversion")
    logger.info("Input table: %s", input_path)
    logger.info("Seed input: %s", seed_input_path)
    logger.info("Aliases file: %s", aliases_path)
    logger.info("Output path: %s", output_path)

    dataframe = load_table(input_path, args.sheet)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    logger.info("Loaded %s rows from nutrition source", len(dataframe))

    name_column = find_column(list(dataframe.columns), NAME_CANDIDATES)
    if not name_column:
        raise SystemExit("Could not find a food-name column in the provided table.")
    logger.info("Detected food-name column: %s", name_column)

    seed_items = load_food_seed(seed_input_path)
    seed_by_slug = {item["slug"]: item for item in seed_items}
    alias_lookup = build_alias_lookup(aliases_path)

    column_map = {
        "calories": find_column(list(dataframe.columns), CALORIES_CANDIDATES),
        "protein": find_column(list(dataframe.columns), PROTEIN_CANDIDATES),
        "carbs": find_column(list(dataframe.columns), CARBS_CANDIDATES),
        "fat": find_column(list(dataframe.columns), FAT_CANDIDATES),
        "fiber": find_column(list(dataframe.columns), FIBER_CANDIDATES),
        "sugar": find_column(list(dataframe.columns), SUGAR_CANDIDATES),
        "sodium": find_column(list(dataframe.columns), SODIUM_CANDIDATES),
        "iron": find_column(list(dataframe.columns), IRON_CANDIDATES),
        "calcium": find_column(list(dataframe.columns), CALCIUM_CANDIDATES),
        "vitamin_c": find_column(list(dataframe.columns), VITC_CANDIDATES),
        "serving": find_column(list(dataframe.columns), SERVING_CANDIDATES),
    }
    logger.debug("Resolved nutrient column map: %s", column_map)

    updated = 0
    created = 0
    unmatched: list[str] = []

    for _, row in dataframe.iterrows():
        raw_name = str(row.get(name_column, "")).strip()
        if not raw_name:
            continue

        slug = alias_lookup.get(normalize(raw_name), slugify(raw_name))
        existing = seed_by_slug.get(slug)

        nutrition = {
            "calories": as_number(row.get(column_map["calories"])) or 0,
            "protein": as_number(row.get(column_map["protein"])) or 0,
            "carbs": as_number(row.get(column_map["carbs"])) or 0,
            "fat": as_number(row.get(column_map["fat"])) or 0,
            "fiber": as_number(row.get(column_map["fiber"])) or 0,
            "sugar": as_number(row.get(column_map["sugar"])) or 0,
            "sodium": as_number(row.get(column_map["sodium"])) or 0,
            "iron": as_number(row.get(column_map["iron"])),
            "calcium": as_number(row.get(column_map["calcium"])),
            "vitamin_c": as_number(row.get(column_map["vitamin_c"])),
        }

        document = {
            "name": existing["name"] if existing else raw_name.title(),
            "slug": slug,
            "category": existing["category"] if existing else DEFAULT_CATEGORY,
            "description": existing["description"] if existing else DEFAULT_DESCRIPTION,
            "default_serving_size_g": (
                as_number(row.get(column_map["serving"]))
                or (existing["default_serving_size_g"] if existing else DEFAULT_SERVING_SIZE)
            ),
            "nutrition_per_100g": nutrition,
            "is_local_food": True if existing is None else existing.get("is_local_food", True),
        }

        validated_document = FoodSeedItem.model_validate(document).model_dump()

        if existing:
            seed_by_slug[slug] = {**existing, **validated_document}
            updated += 1
        else:
            seed_by_slug[slug] = validated_document
            created += 1
            unmatched.append(raw_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_items = sorted(seed_by_slug.values(), key=lambda item: item["name"])
    output_path.write_text(json.dumps(ordered_items, indent=2), encoding="utf-8")

    logger.info("Converted nutrition table from %s", input_path)
    logger.info("Updated existing FoodIntel foods: %s", updated)
    logger.info("Created new FoodIntel foods: %s", created)
    logger.info("Output written to: %s", output_path)
    if unmatched:
        logger.warning("New foods created from unmatched rows (showing up to 30):")
        for name in unmatched[:30]:
            logger.warning("  - %s", name)


if __name__ == "__main__":
    main()
