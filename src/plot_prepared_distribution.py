"""
plot_prepared_distribution.py — bar chart of the PREPARED 4-class dataset.
============================================================================
Run from the project root:

    .venv\\Scripts\\python src\\plot_prepared_distribution.py

Scans data/processed/{train,val,test}/<class>/ (the actual files the model
trains on), cross-checks the totals against data/splits/split_info.json, and
saves outputs/figures/prepared_distribution.png — the chart that belongs in
the final report (Section 6 + Section 10).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SPLITS = ("train", "val", "test")


def main(show: bool = False) -> None:
    if not show:
        matplotlib.use("Agg")
    classes = sorted(p.name for p in (PROCESSED_DIR / "train").iterdir()
                     if p.is_dir())
    counts = {s: [len(list((PROCESSED_DIR / s / c).glob("*.*")))
                  for c in classes] for s in SPLITS}

    # Cross-check against the split manifest record.
    import json
    info = json.loads(
        (PROJECT_ROOT / "data" / "splits" / "split_info.json").read_text())
    for c in classes:
        for s in SPLITS:
            assert counts[s][classes.index(c)] == info["class_counts"][c][s], \
                f"disk != manifest for {s}/{c}"
    print("Disk scan matches split_info.json manifest.")

    x = range(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, s in enumerate(SPLITS):
        bars = ax.bar([p + (i - 1) * width for p in x], counts[s],
                      width=width, label=s)
        for bar, v in zip(bars, counts[s]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 12, str(v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x), classes)
    ax.set_xlabel("Waste category (prepared 4-class dataset)")
    ax.set_ylabel("Number of images")
    ax.set_title("Prepared dataset: images per class and split (total 2,388)")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "prepared_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    totals = {c: sum(counts[s][i] for s in SPLITS)
              for i, c in enumerate(classes)}
    print(totals, "total:", sum(totals.values()))
    print(f"Saved to: {out}\nDone.")


if __name__ == "__main__":
    import sys

    main(show="--show" in sys.argv)
