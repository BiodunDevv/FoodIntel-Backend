import json
from pathlib import Path
from typing import Any

from app.config.settings import BACKEND_DIR, ROOT_DIR
from app.schemas.food_schema import FoodSeedItem


DEFAULT_FOOD_SEED_PATH = BACKEND_DIR / "app/data/food_nutrition_seed.json"
DEFAULT_FOOD_ALIASES_PATH = BACKEND_DIR / "app/data/food_slug_aliases.json"


def resolve_project_path(path_value: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve a path safely from cwd, repo root, or backend root."""
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        if must_exist and not raw_path.exists():
            raise FileNotFoundError(f"Path does not exist: {raw_path}")
        return raw_path.resolve()

    candidates = [
        (Path.cwd() / raw_path).resolve(),
        (ROOT_DIR / raw_path).resolve(),
        (BACKEND_DIR / raw_path).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if must_exist:
        checked = " | ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Path does not exist for '{path_value}'. Checked: {checked}"
        )

    return (ROOT_DIR / raw_path).resolve()


def load_food_seed(seed_path: str | Path | None = None) -> list[dict[str, Any]]:
    resolved_seed_path = (
        resolve_project_path(seed_path, must_exist=True)
        if seed_path is not None
        else DEFAULT_FOOD_SEED_PATH
    )
    payload = json.loads(resolved_seed_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Food nutrition seed JSON must be a list of objects.")

    validated: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid seed item at index {index}: expected an object.")
        model = FoodSeedItem.model_validate(item)
        validated.append(model.model_dump())
    return validated


def load_slug_aliases(path_value: str | Path | None = None) -> dict[str, list[str]]:
    resolved_alias_path = (
        resolve_project_path(path_value, must_exist=True)
        if path_value is not None
        else DEFAULT_FOOD_ALIASES_PATH
    )
    payload = json.loads(resolved_alias_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Slug aliases JSON must be an object keyed by slug.")

    normalized: dict[str, list[str]] = {}
    for slug, aliases in payload.items():
        if not isinstance(slug, str):
            raise ValueError("Alias mapping contains a non-string slug key.")
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ValueError(f"Alias mapping for '{slug}' must be a list of strings.")
        normalized[slug] = aliases
    return normalized
