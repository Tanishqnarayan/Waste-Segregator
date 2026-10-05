"""
explore_dataset.py — PHASE 2: Dataset exploration.
==================================================
Run from the project root:

    .venv\\Scripts\\python src\\explore_dataset.py            # save figures only
    .venv\\Scripts\\python src\\explore_dataset.py --show    # also pop up windows

What this script teaches / does:
  1. Finds the dataset with pathlib, relative to THIS file — no hard-coded
     absolute paths (like C:\\Users\\...) so it runs on any machine.
  2. Lists the category sub-folders (TrashNet: cardboard, glass, metal,
     paper, plastic, trash).
  3. Opens every image with Pillow to prove it is readable, and counts the
     valid images per class. Unreadable files are reported, not hidden.
  4. Prints a per-class count table with percentages + the grand total.
  5. Saves and shows a bar chart of the class distribution.
  6. Saves and shows one sample image from each category.

Figures are saved under outputs/figures/ (created automatically) BEFORE any
plot window opens, so the files exist even if you close the windows quickly.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# --- Paths (relative, portable) -------------------------------------------
# __file__ = this script. parents[1] = project root (src/ -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "dataset-resized"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Image file types we accept (lower-cased suffix match).
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_class_folders(data_dir: Path) -> list[Path]:
    """Return sorted sub-folders of the dataset dir, or raise a clear error."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            f"Expected it at: <project-root>/data/dataset-resized\n"
            "Fix: copy the extracted 'dataset-resized' folder there and re-run."
        )
    folders = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not folders:
        raise FileNotFoundError(
            f"No category sub-folders inside {data_dir}. "
            "Expected folders like cardboard/, glass/, metal/, paper/, ..."
        )
    return folders


def count_valid_images(class_dir: Path) -> tuple[int, list[str], Path | None]:
    """Count readable images in one class folder.

    Returns (valid_count, unreadable_filenames, first_valid_image_path).
    An image is 'valid' only if Pillow can fully decode it (verify + load).
    """
    valid = 0
    unreadable: list[str] = []
    first_valid: Path | None = None

    for path in sorted(class_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue  # skip .DS_Store, .txt, hidden files, etc.
        try:
            with Image.open(path) as img:
                img.verify()  # checks the file decodes, without loading pixels
            with Image.open(path) as img:
                img.load()  # actually load pixels (catches truncated files)
            valid += 1
            if first_valid is None:
                first_valid = path
        except Exception:  # noqa: BLE001 — any decode failure means "unreadable"
            unreadable.append(path.name)
    return valid, unreadable, first_valid


def plot_distribution(counts: dict[str, int]) -> Path:
    """Bar chart of images per class. Returns the saved figure path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "class_distribution.png"

    names = sorted(counts)
    values = [counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color="seagreen")
    ax.set_title("TrashNet (dataset-resized): images per class")
    ax.set_xlabel("Waste category")
    ax.set_ylabel("Number of images")
    # Print the exact count on top of each bar.
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value),
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_samples(samples: dict[str, Path]) -> Path:
    """Grid with one sample image per class. Returns the saved figure path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "sample_images.png"

    names = sorted(samples)
    n = len(names)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten()
    for ax, name in zip(axes, names):
        with Image.open(samples[name]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{name}\n({samples[name].name})", fontsize=10)
        ax.axis("off")
    # Hide any leftover empty subplot (e.g. 6 images in a 2x3 grid = exact fit).
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Sample image from each TrashNet category", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def main(show: bool = False) -> None:
    if not show:
        # Non-interactive backend: render figures to files without ever
        # opening a window (safe for automation, servers, background runs).
        matplotlib.use("Agg")
    print(f"Looking for dataset at: {DATA_DIR}")
    class_folders = find_class_folders(DATA_DIR)
    print(f"Found {len(class_folders)} category folders: "
          f"{', '.join(p.name for p in class_folders)}\n")

    counts: dict[str, int] = {}
    samples: dict[str, Path] = {}
    all_unreadable: dict[str, list[str]] = {}

    for folder in class_folders:
        valid, unreadable, first = count_valid_images(folder)
        counts[folder.name] = valid
        if first is not None:
            samples[folder.name] = first
        if unreadable:
            all_unreadable[folder.name] = unreadable

    total = sum(counts.values())

    # --- Count table -------------------------------------------------------
    print(f"{'Class':<12} {'Images':>7} {'Share':>7}")
    print("-" * 28)
    for name in sorted(counts):
        share = 100 * counts[name] / total if total else 0
        print(f"{name:<12} {counts[name]:>7} {share:>6.1f}%")
    print("-" * 28)
    print(f"{'TOTAL':<12} {total:>7}\n")

    if all_unreadable:
        print("WARNING — unreadable files (excluded from counts):")
        for name, files in all_unreadable.items():
            print(f"  {name}: {len(files)} file(s), e.g. {files[:3]}")
    else:
        print("All counted images opened successfully (0 unreadable files).")

    # --- Figures (saved first, shown after) --------------------------------
    dist_path = plot_distribution(counts)
    print(f"\nBar chart saved to: {dist_path}")
    if samples:
        sample_path = plot_samples(samples)
        print(f"Sample grid saved to: {sample_path}")

    plt.show() if show else plt.close("all")  # pop-ups only with --show
    print("\nDone." + (" Close the plot windows to exit." if show else ""))


if __name__ == "__main__":
    import sys

    main(show="--show" in sys.argv)
