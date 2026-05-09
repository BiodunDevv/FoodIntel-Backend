"""Download-aware dataset merger for FoodIntel Phase 1.

This script does not require network access. It expects dataset archives or
extracted dataset folders to exist under a raw data root, then normalizes class
names and copies supported images into one ImageFolder-style directory:
``ml/data/raw/unified/{canonical_class_name}/``.

The same script is used by the Colab notebook after dataset download cells have
placed files under ``/content/data/raw``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_ARCHIVES = {".zip"}
LOW_DATA_THRESHOLD = 100
SKIP_CLASSES = {"skip", "ignore", "unknown", "bread"}


CANONICAL_NAMES: dict[str, str] = {
    # Existing FoodIntel Nigerian aliases.
    "abacha": "abacha",
    "abacha and ugba african salad": "abacha",
    "abacha and ugba(african salad)": "abacha",
    "afang soup": "afang_soup",
    "akara": "akara",
    "akarabread": "akara",
    "akara bread": "akara",
    "akara and eko": "akara",
    "akara and eko akamu": "akara",
    "akara and eko-akamu": "akara",
    "amala and ewedu gbegiri": "amala",
    "amala and ewedu-gbegiri": "amala",
    "amala and gbegiri ewedu": "amala",
    "amala and gbegiri- ewedu": "amala",
    "asaro": "asaro",
    "beans": "beans",
    "boli": "boli",
    "boli bole": "boli",
    "boli(bole)": "boli",
    "chin chin": "chin_chin",
    "draw soup": "draw_soup",
    "egusi": "egusi_soup",
    "egusi soup": "egusi_soup",
    "egusi": "egusi_soup",
    "ewang": "ekwang",
    "ekwang": "ekwang",
    "eru": "eru",
    "ewa agoyin": "beans",
    "ewa-agoyin": "beans",
    "ewedu soup": "ewedu_soup",
    "ewedu": "ewedu_soup",
    "fried plantain": "plantain",
    "fried plantains dodo": "plantain",
    "fried plantains (dodo)": "plantain",
    "garri and groundnut": "soaking_garri",
    "garriandgrounut": "soaking_garri",
    "jellof": "jollof_rice_nigeria",
    "jollof": "jollof_rice_nigeria",
    "kilishi": "kilishi",
    "meat pie": "meat_pie",
    "meat-pie": "meat_pie",
    "moi moi": "moi_moi",
    "moi-moi": "moi_moi",
    "moimoi": "moi_moi",
    "moin moin": "moi_moi",
    "moin-moin": "moi_moi",
    "ndole": "ndole",
    "nkwobi": "nkwobi",
    "nigerian akara": "akara",
    "nigerian draw soup": "draw_soup",
    "nigerian egusi soup": "egusi_soup",
    "nigerian ewedu soup": "ewedu_soup",
    "nigerian moi moi": "moi_moi",
    "nigerian nkwobi": "nkwobi",
    "nigerian noodles": "noodles",
    "nigerian nsala soup white soup": "ofe_nsala",
    "nigerian nsala soup - white soup": "ofe_nsala",
    "nigerian ora soup": "ora_soup",
    "nigerian rice and tomatoe stew": "rice_and_stew",
    "nigerian rice and tomato stew": "rice_and_stew",
    "nigerian suya": "suya",
    "nigerian yam porridge": "yam_porridge",
    "nigerian zobo": "zobo",
    "noodles": "noodles",
    "nsala soup": "ofe_nsala",
    "ofe akwu": "ofe_akwu",
    "ofe nsala": "ofe_nsala",
    "ofeowerri": "ofe_owerri",
    "ofe owerri": "ofe_owerri",
    "ogbono": "ogbono_soup",
    "oha soup": "oha_soup",
    "oha_soup": "oha_soup",
    "okra": "okro_soup",
    "okro soup": "okro_soup",
    "ora soup": "ora_soup",
    "palm oil soup": "ofe_akwu",
    "palm nut soup": "palm_nut_soup",
    "palm-nut soup": "palm_nut_soup",
    "pepper soup": "pepper_soup",
    "pepper-soup": "pepper_soup",
    "plantain": "plantain",
    "puff puff": "puff_puff",
    "puff-puff": "puff_puff",
    "pufpuf": "puff_puff",
    "rice and stew": "rice_and_stew",
    "shawarma": "shawarma",
    "soaking garri naija manpower": "soaking_garri",
    "soaking garri - naija manpower": "soaking_garri",
    "soaking garri": "soaking_garri",
    "suya": "suya",
    "vegetable soup": "vegetable_soup",
    "waakye": "waakye",
    "yam": "yam",
    "yam porridge": "yam_porridge",
    "zobo": "zobo",
    # Jollof variants that must stay distinct where provenance says so.
    "ghana jollof": "jollof_rice_ghana",
    "jollof ghana": "jollof_rice_ghana",
    "jollof rice ghana": "jollof_rice_ghana",
    "jollof-ghana": "jollof_rice_ghana",
    "jollof_ghana": "jollof_rice_ghana",
    "jollof nigeria": "jollof_rice_nigeria",
    "jollof rice nigeria": "jollof_rice_nigeria",
    "jollof-nigeria": "jollof_rice_nigeria",
    "jollof_nigeria": "jollof_rice_nigeria",
    "nigeria jollof": "jollof_rice_nigeria",
    "nigeria_jollof": "jollof_rice_nigeria",
    # Generic jollof defaults to Nigerian in this project unless a source says Ghana.
    "jollof rice": "jollof_rice_nigeria",
    "jollof_rice": "jollof_rice_nigeria",
    # Food-101 baseline classes.
    "fried rice": "fried_rice",
    "fried_rice": "fried_rice",
    "hamburger": "hamburger",
    "pizza": "pizza",
    "soup": "soup",
    "sushi": "sushi",
}


@dataclass(frozen=True)
class Source:
    """A raw dataset source to scan."""

    name: str
    path: Path


@dataclass(frozen=True)
class ImageRecord:
    """A discovered image and its normalized metadata."""

    source_dataset: str
    original_path: Path
    raw_class_name: str
    canonical_class_name: str
    sha256_hash: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    default_raw_root = os.getenv("FOODINTEL_RAW_ROOT", "ml/data/raw")
    parser = argparse.ArgumentParser(
        description="Merge FoodIntel raw datasets into ml/data/raw/unified with a manifest."
    )
    parser.add_argument(
        "--raw-root",
        default=default_raw_root,
        help="Root containing extracted raw datasets. Defaults to FOODINTEL_RAW_ROOT or ml/data/raw.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("FOODINTEL_UNIFIED_DIR", f"{default_raw_root}/unified"),
        help="Unified ImageFolder output directory.",
    )
    parser.add_argument(
        "--manifest-path",
        default=os.getenv("FOODINTEL_MANIFEST_PATH", f"{default_raw_root}/dataset_manifest.json"),
        help="Path for the per-image dataset manifest JSON.",
    )
    parser.add_argument(
        "--class-map",
        default=os.getenv("FOODINTEL_CLASS_MAP", "ml/nigerian_class_map.extensive.json"),
        help="Optional JSON alias map to merge with built-in aliases.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Additional source folder to scan. May be repeated. If NAME= is omitted, "
            "the folder name is used."
        ),
    )
    parser.add_argument(
        "--no-default-sources",
        action="store_true",
        help="Only scan --source entries instead of the standard folders under --raw-root.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the unified output directory before copying images.",
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="PATH",
        help="Archive to copy into --archive-dir and extract before scanning. May be repeated.",
    )
    parser.add_argument(
        "--archive-search-root",
        action="append",
        default=[],
        metavar="PATH",
        help="Folder to search recursively for local archives. May be repeated.",
    )
    parser.add_argument(
        "--archive-dir",
        default=os.getenv("FOODINTEL_ARCHIVE_DIR", "ml/raw_sources/archives"),
        help="Folder where local archive files are centralized.",
    )
    parser.add_argument(
        "--extract-dir",
        default=os.getenv("FOODINTEL_EXTRACT_DIR", f"{default_raw_root}/extracted"),
        help="Folder where archives are extracted before merging.",
    )
    parser.add_argument(
        "--clear-extracted",
        action="store_true",
        help="Delete --extract-dir before extracting archives.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    """Normalize a class or folder name for alias lookup.

    Args:
        value: Raw class, folder, or file stem.

    Returns:
        A lowercase underscore-separated name.
    """

    cleaned = value.strip().lower()
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.replace(" ", "_")


def alias_key(value: str) -> str:
    """Normalize a class name into the alias dictionary key style."""

    return normalize_name(value).replace("_", " ")


def load_aliases(class_map_path: Path) -> dict[str, str]:
    """Load canonical class aliases from built-in values and an optional JSON map."""

    aliases = {alias_key(key): value for key, value in CANONICAL_NAMES.items()}
    if class_map_path.exists():
        payload = json.loads(class_map_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Class map must be a JSON object: {class_map_path}")
        aliases.update({alias_key(str(key)): normalize_name(str(value)) for key, value in payload.items()})
    return aliases


def canonicalize_class_name(raw_name: str, aliases: dict[str, str]) -> str:
    """Return the canonical class name for a raw class label."""

    key = alias_key(raw_name)
    return aliases.get(key, normalize_name(raw_name))


def parse_source(raw_source: str) -> Source:
    """Parse a CLI source entry into a named source."""

    if "=" in raw_source:
        name, path = raw_source.split("=", 1)
        return Source(name=name.strip() or Path(path).name, path=Path(path))
    path = Path(raw_source)
    return Source(name=path.name, path=path)


def default_sources(raw_root: Path) -> list[Source]:
    """Return the standard Phase 1 source folders beneath a raw root."""

    return [
        Source("mendeley_african_foods", raw_root / "mendeley"),
        Source("kaggle_nigeria_food_ai", raw_root / "kaggle_nigeria"),
        Source("foodnet", raw_root / "foodnet"),
        Source("food101_subset", raw_root / "food101"),
        Source("app_feedback", raw_root / "feedback"),
    ]


def archive_suffix(path: Path) -> str:
    """Return a normalized archive suffix."""

    name = path.name.lower()
    if name.endswith(".zip"):
        return ".zip"
    return path.suffix.lower()


def is_supported_archive(path: Path) -> bool:
    """Return whether a path is a supported archive."""

    return path.is_file() and archive_suffix(path) in SUPPORTED_ARCHIVES


def safe_archive_name(path: Path) -> str:
    """Return a stable archive filename safe for the centralized archive folder."""

    stem = normalize_name(path.stem) or "archive"
    suffix = archive_suffix(path)
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}{suffix}"


def find_archives(search_roots: Iterable[Path]) -> list[Path]:
    """Find local supported archives under the provided roots."""

    archives: list[Path] = []
    ignored_parts = {".git", ".venv", "__pycache__", "node_modules"}
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and is_supported_archive(root):
            archives.append(root)
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if any(part in ignored_parts for part in path.parts):
                continue
            if is_supported_archive(path):
                archives.append(path)
    unique: dict[Path, Path] = {path.resolve(): path for path in archives}
    return sorted(unique.values())


def centralize_archives(archives: Iterable[Path], archive_dir: Path) -> list[Path]:
    """Copy archives into one source archive folder without deleting originals."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for archive in archives:
        if not archive.exists():
            continue
        destination = archive_dir / archive.name
        if destination.exists() and destination.resolve() != archive.resolve():
            destination = archive_dir / safe_archive_name(archive)
        if destination.resolve() != archive.resolve():
            shutil.copy2(archive, destination)
        staged.append(destination)
    return sorted({path.resolve(): path for path in staged}.values())


def extract_archive(archive: Path, extract_dir: Path) -> Source | None:
    """Extract a supported archive and return its scan source."""

    target = extract_dir / normalize_name(archive.stem)
    if target.exists() and any(target.iterdir()):
        return Source(normalize_name(archive.stem), target)

    target.mkdir(parents=True, exist_ok=True)
    if archive_suffix(archive) == ".zip":
        try:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
        except zipfile.BadZipFile:
            print(f"Skipping invalid ZIP archive: {archive}")
            return None
        return Source(normalize_name(archive.stem), target)

    return None


def extract_archives(archives: Iterable[Path], extract_dir: Path, clear_extracted: bool) -> list[Source]:
    """Extract all archives and return scan sources."""

    if clear_extracted and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    sources: list[Source] = []
    for archive in archives:
        source = extract_archive(archive, extract_dir)
        if source is not None:
            sources.append(source)
    return sources


def iter_image_paths(source: Source, output_dir: Path) -> Iterable[Path]:
    """Yield supported image paths from a source, excluding the output folder."""

    output_dir = output_dir.resolve()
    for path in sorted(source.path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(output_dir)
            continue
        except ValueError:
            yield path


def infer_class_name(image_path: Path, source_root: Path) -> str:
    """Infer a class name from an image path.

    The nearest parent directory is the common ImageFolder convention and works
    for class folders nested under train/val/test or dataset-specific wrappers.
    """

    parent = image_path.parent
    if parent == source_root:
        return image_path.stem
    if parent.name.lower() in {"train", "valid", "val", "test"}:
        return "skip"
    return parent.name


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(
    sources: list[Source], output_dir: Path, aliases: dict[str, str]
) -> tuple[list[ImageRecord], list[Source]]:
    """Discover images and return unavailable sources separately."""

    records: list[ImageRecord] = []
    unavailable: list[Source] = []
    seen_paths: set[Path] = set()
    for source in sources:
        if not source.path.exists():
            unavailable.append(source)
            continue
        for image_path in iter_image_paths(source, output_dir):
            resolved_path = image_path.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            raw_class_name = infer_class_name(image_path, source.path)
            canonical_class_name = canonicalize_class_name(raw_class_name, aliases)
            if canonical_class_name.lower() in SKIP_CLASSES:
                continue
            records.append(
                ImageRecord(
                    source_dataset=source.name,
                    original_path=image_path,
                    raw_class_name=raw_class_name,
                    canonical_class_name=canonical_class_name,
                    sha256_hash=sha256_file(image_path),
                )
            )
    return records, unavailable


def safe_destination_name(record: ImageRecord) -> str:
    """Build a stable, collision-resistant destination filename."""

    source_prefix = normalize_name(record.source_dataset)
    original_stem = normalize_name(record.original_path.stem)[:80] or "image"
    suffix = record.original_path.suffix.lower()
    return f"{source_prefix}_{record.sha256_hash[:12]}_{original_stem}{suffix}"


def copy_records(records: list[ImageRecord], output_dir: Path) -> list[dict[str, str]]:
    """Copy discovered images into the unified directory and build manifest rows."""

    manifest: list[dict[str, str]] = []
    seen_hashes: set[tuple[str, str]] = set()
    for record in records:
        dedupe_key = (record.canonical_class_name, record.sha256_hash)
        if dedupe_key in seen_hashes:
            continue
        seen_hashes.add(dedupe_key)
        class_dir = output_dir / record.canonical_class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / safe_destination_name(record)
        if destination.exists():
            destination = class_dir / f"{destination.stem}_{len(manifest) + 1}{destination.suffix}"
        shutil.copy2(record.original_path, destination)
        manifest.append(
            {
                "filename": destination.name,
                "class": record.canonical_class_name,
                "source_dataset": record.source_dataset,
                "original_path": str(record.original_path),
                "raw_class_name": record.raw_class_name,
                "sha256_hash": record.sha256_hash,
            }
        )
    return manifest


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    """Write the dataset manifest JSON."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def print_summary(rows: list[dict[str, str]], unavailable: list[Source]) -> None:
    """Print a class/source summary table."""

    source_counts: Counter[tuple[str, str]] = Counter(
        (row["class"], row["source_dataset"]) for row in rows
    )
    class_totals: Counter[str] = Counter(row["class"] for row in rows)

    if unavailable:
        print("Unavailable sources:")
        for source in unavailable:
            print(f"  {source.name}: {source.path}")
        print()

    print("class | source | count | total | status")
    print("-" * 72)
    for class_name in sorted(class_totals):
        status = "LOW DATA" if class_totals[class_name] < LOW_DATA_THRESHOLD else ""
        for source_name in sorted(
            source for (candidate_class, source) in source_counts if candidate_class == class_name
        ):
            count = source_counts[(class_name, source_name)]
            print(f"{class_name} | {source_name} | {count} | {class_totals[class_name]} | {status}")

    print()
    print(f"Total images copied: {len(rows)}")
    print(f"Total classes: {len(class_totals)}")


def main() -> None:
    """Run the dataset merge workflow."""

    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    archive_dir = Path(args.archive_dir)
    extract_dir = Path(args.extract_dir)
    aliases = load_aliases(Path(args.class_map))

    archive_candidates = [Path(archive) for archive in args.archive]
    archive_candidates.extend(find_archives(Path(root) for root in args.archive_search_root))
    staged_archives = centralize_archives(archive_candidates, archive_dir)
    extracted_sources = extract_archives(staged_archives, extract_dir, args.clear_extracted)

    sources = [] if args.no_default_sources else default_sources(raw_root)
    sources.extend(extracted_sources)
    sources.extend(parse_source(source) for source in args.source)

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, unavailable = discover_images(sources, output_dir, aliases)
    manifest = copy_records(records, output_dir)
    write_manifest(manifest_path, manifest)
    print_summary(manifest, unavailable)
    print(f"Unified dataset: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
