"""
data_preparation.py — PHASE 3: Build the 4-class dataset.
==========================================================
Run from the project root (organic images must be downloaded first — see
"STEP 0" below):

    .venv\\Scripts\\python src\\data_preparation.py --organic-src data\\organic-raw

What it does:
  1. Maps TrashNet folders to our 4 target classes:
         plastic -> plastic            (482 images)
         paper + cardboard -> paper    (594 + 403 = 997 images)
         metal -> metal                (410 images)
         <organic folder> -> organic   (we take up to --organic-limit)
     Glass and trash are NOT used — glass is not organic, trash is mixed junk.
  2. Drops exact-duplicate files (same MD5 bytes) so copies of one photo
     cannot leak across splits.
  3. Stratified 70/15/15 train/val/test split per class (fixed --seed, so the
     split is reproducible — same seed always gives the same split).
  4. COPIES (never moves) images into data/processed/{train,val,test}/<class>/.
     Originals are never touched.
  5. Saves models/class_names.json (alphabetical — the same order Keras uses),
     data/splits/manifest.csv (every file + its split) and
     data/splits/split_info.json (seed, ratios, counts) for reproducibility.

STEP 0 — get the organic images (you do this once, in your browser):
  1. Go to https://data.mendeley.com/datasets/n3gtgm9jxj/3  (free account needed)
     "Waste Classification Dataset" — 13,880 organic + 10,825 recyclable photos,
     licence CC BY 4.0 (cite Nnamoko et al. 2022 — see README/report later).
  2. Click "Download All", unzip, and copy ONLY the organic folder's images
     into <project-root>/data/organic-raw/ (flat is fine, sub-folders are fine).
  3. Then run this script with --organic-src data/organic-raw
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

# --- Paths (relative, portable) -------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRASHNET_DIR = PROJECT_ROOT / "data" / "dataset-resized"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CLASS_NAMES_JSON = PROJECT_ROOT / "models" / "class_names.json"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# TrashNet folder -> our class. Folders NOT listed here (glass, trash) are
# deliberately excluded: neither is organic waste.
TRASHNET_CLASS_MAP = {
    "plastic": "plastic",
    "paper": "paper",
    "cardboard": "paper",  # same material family as paper; keeps labels clean
    "metal": "metal",
}


def md5_of(path: Path) -> str:
    """Fingerprint a file's bytes (used to find exact duplicates)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images(folder: Path, recursive: bool = False) -> list[Path]:
    """All image files in a folder (sorted). Recursive for the organic dump."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder not found: {folder}\n"
            "For --organic-src: download the Mendeley set (see STEP 0 in this "
            "file's docstring) and point to the folder holding organic images."
        )
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def dedupe(paths: list[Path]) -> tuple[list[Path], int]:
    """Drop exact-duplicate files (identical bytes), keep first alphabetically."""
    seen: dict[str, Path] = {}
    for p in paths:
        seen.setdefault(md5_of(p), p)
    unique = sorted(seen.values())
    return unique, len(paths) - len(unique)


def split_paths(paths: list[Path], ratios: tuple[float, float, float],
                seed: int) -> dict[str, list[Path]]:
    """Stratified-by-construction split: shuffle once, slice 70/15/15."""
    train_ratio, val_ratio, _ = ratios
    train, rest = train_test_split(paths, train_size=train_ratio,
                                   random_state=seed, shuffle=True)
    val_share_of_rest = val_ratio / (1.0 - train_ratio)
    val, test = train_test_split(rest, train_size=val_share_of_rest,
                                  random_state=seed, shuffle=True)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def main(organic_src: Path, organic_limit: int, ratios: tuple[float, float, float],
         seed: int, output_dir: Path, splits_dir: Path,
         class_names_json: Path) -> None:
    # --- 1. Gather per-class file lists ------------------------------------
    class_files: dict[str, list[Path]] = {}
    for folder, cls in TRASHNET_CLASS_MAP.items():
        class_files.setdefault(cls, []).extend(collect_images(TRASHNET_DIR / folder))

    organic_all = collect_images(organic_src, recursive=True)
    if not organic_all:
        raise FileNotFoundError(f"No images found under {organic_src}")
    # Deterministic subset: sorted names + fixed limit (reproducible).
    organic_kept = organic_all[:organic_limit]
    print(f"Organic pool: {len(organic_all)} images found, using first "
          f"{len(organic_kept)} (sorted order, --organic-limit={organic_limit})")
    class_files["organic"] = organic_kept

    # --- 2. Drop exact duplicates (anti-leakage) ----------------------------
    dup_total = 0
    for cls, paths in class_files.items():
        unique, n_dup = dedupe(sorted(paths))
        class_files[cls] = unique
        dup_total += n_dup
        if n_dup:
            print(f"  {cls}: removed {n_dup} exact-duplicate file(s)")
    print(f"Exact duplicates removed: {dup_total}\n")

    # --- 3. Split, copy, manifest -------------------------------------------
    class_names = sorted(class_files)  # alphabetical == Keras class order
    manifest_rows: list[tuple[str, str, str, str]] = []  # split, class, file, src
    summary: dict[str, dict[str, int]] = {}

    for cls in class_names:
        parts = split_paths(class_files[cls], ratios, seed)
        summary[cls] = {s: len(p) for s, p in parts.items()}
        for split, paths in parts.items():
            dest_dir = output_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                shutil.copy2(src, dest_dir / src.name)  # copy, never move
                manifest_rows.append((split, cls, src.name, str(src)))

    # --- 4. Save reproducibility artefacts ----------------------------------
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [("split", "class", "filename", "source_path"), *manifest_rows]
        )
    split_info = {
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "organic_limit": organic_limit,
        "organic_source": str(organic_src),
        "class_counts": summary,
        "total": sum(sum(v.values()) for v in summary.values()),
    }
    with open(splits_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    class_names_json.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # --- 5. Report ------------------------------------------------------------
    print(f"{'Class':<10} {'train':>6} {'val':>6} {'test':>6} {'total':>7}")
    print("-" * 39)
    for cls in class_names:
        s = summary[cls]
        print(f"{cls:<10} {s['train']:>6} {s['val']:>6} {s['test']:>6} "
              f"{sum(s.values()):>7}")
    print("-" * 39)
    print(f"{'TOTAL':<10} "
          f"{sum(v['train'] for v in summary.values()):>6} "
          f"{sum(v['val'] for v in summary.values()):>6} "
          f"{sum(v['test'] for v in summary.values()):>6} "
          f"{split_info['total']:>7}")
    print(f"\nCopied dataset -> {output_dir}")
    print(f"Manifest        -> {splits_dir / 'manifest.csv'}")
    print(f"Split info      -> {splits_dir / 'split_info.json'}")
    print(f"Class names     -> {class_names_json}  {class_names}")
    print("\nDone. Originals untouched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the 4-class waste dataset.")
    parser.add_argument("--organic-src", type=Path, required=True,
                        help="Folder holding organic-waste images (see STEP 0).")
    parser.add_argument("--organic-limit", type=int, default=500,
                        help="Max organic images to use (default 500, for balance).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the split (default 42).")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR,
                        help="Where to write the split dataset (advanced/testing).")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--class-names-json", type=Path, default=CLASS_NAMES_JSON)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-9:
        sys.exit("ERROR: --train + --val + --test must sum to 1.0")
    main(args.organic_src, args.organic_limit,
         (args.train, args.val, args.test), args.seed,
         args.output_dir, args.splits_dir, args.class_names_json)
