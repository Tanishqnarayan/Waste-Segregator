"""
evaluate.py — PHASE 6: Honest scoring on the SEALED test set.
==============================================================
Run from the project root:

    .venv\\Scripts\\python src\\evaluate.py

What it does:
  1. Loads the test split (never seen in training) via preprocessing.py.
  2. Scores BOTH models: stage_a_best.keras (frozen base) and
     waste_classifier.keras (fine-tuned) — so you can see what fine-tuning
     actually bought you.
  3. Reports accuracy, per-class precision/recall/F1, macro-F1, confusion
     matrix; saves training curves + confusion-matrix figures.
  4. Saves everything to outputs/evaluation.json + outputs/figures/.

Reading the metrics (the 30-second version):
  - accuracy = fraction of test photos labeled correctly. Easy but hides
    weak classes when classes are imbalanced.
  - precision (per class) = "when the model SAYS paper, how often is it
    right?" Low precision = false alarms.
  - recall (per class) = "of all REAL paper photos, how many did it catch?"
    Low recall = misses.
  - F1 = balance of the two. macro-F1 = plain average over classes: the
    fairest single number when classes are imbalanced (paper can't hide
    behind its size).
  - confusion matrix: rows = true class, columns = predicted class. The
    diagonal is correct; off-diagonal blobs show which classes get confused.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_datasets  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "stage_a (frozen base)": PROJECT_ROOT / "models" / "stage_a_best.keras",
    "final (fine-tuned)": PROJECT_ROOT / "models" / "waste_classifier.keras",
}
HISTORIES = {
    "stage_a (frozen base)": PROJECT_ROOT / "outputs" / "history_stage_a.json",
    "final (fine-tuned)": PROJECT_ROOT / "outputs" / "history_stage_b.json",
}
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def score_model(model: tf.keras.Model, test_ds: tf.data.Dataset,
                class_names: list[str]) -> dict:
    """Run the model over test, return metrics dict + raw predictions."""
    y_true, y_pred = [], []
    for x, y in test_ds:
        y_true.extend(y.numpy().tolist())
        y_pred.extend(model.predict(x, verbose=0).argmax(axis=1).tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    per_class = {c: {m: round(float(report[c][m]), 4)
                     for m in ("precision", "recall", "f1-score")}
                 for c in class_names}
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
        "report_text": classification_report(y_true, y_pred,
                                             target_names=class_names,
                                             zero_division=0),
        "_cm": cm,  # kept for plotting, stripped before saving JSON
    }


def plot_curves() -> Path:
    """Train/val accuracy + loss across Stage A then Stage B (continuous x)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "training_curves.png"
    ha = json.loads(HISTORIES["stage_a (frozen base)"].read_text())
    hb = json.loads(HISTORIES["final (fine-tuned)"].read_text())

    def cat(key):
        return ha.get(key, []) + hb.get(key, [])

    epochs = range(1, len(cat("accuracy")) + 1)
    split_at = len(ha.get("accuracy", []))  # where Stage B begins
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, train_k, val_k, title in [
            (ax1, "accuracy", "val_accuracy", "Accuracy"),
            (ax2, "loss", "val_loss", "Loss")]:
        ax.plot(epochs, cat(train_k), label=f"train {title.lower()}")
        ax.plot(epochs, cat(val_k), label=f"val {title.lower()}")
        if split_at:
            ax.axvline(split_at + 0.5, color="gray", linestyle="--",
                       label="fine-tune starts")
        ax.set_xlabel("epoch (A then B)")
        ax.set_title(f"Training vs validation {title.lower()}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion(cm: np.ndarray, class_names: list[str]) -> Path:
    out = FIGURES_DIR / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    fig.colorbar(im, ax=ax, label="test images")
    ax.set_xticks(range(len(class_names)), class_names, rotation=20)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix — final model, sealed test set")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(show: bool = False) -> None:
    if not show:
        matplotlib.use("Agg")
    class_names = json.loads(
        (PROJECT_ROOT / "models" / "class_names.json").read_text())
    _, _, test_ds, folder_order = load_datasets()
    assert folder_order == class_names

    results: dict[str, dict] = {}
    for label, path in MODELS.items():
        print(f"Scoring {label} ...")
        model = tf.keras.models.load_model(path)
        results[label] = score_model(model, test_ds, class_names)
        r = results[label]
        print(f"  test accuracy {r['accuracy']:.4f}   macro-F1 {r['macro_f1']:.4f}")
    print("\n" + results["final (fine-tuned)"]["report_text"])

    curves = plot_curves()
    cm_path = plot_confusion(results["final (fine-tuned)"]["_cm"], class_names)
    print(f"Curves saved to: {curves}")
    print(f"Confusion matrix saved to: {cm_path}")

    serializable = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                    for k, v in results.items()}
    eval_path = PROJECT_ROOT / "outputs" / "evaluation.json"
    eval_path.write_text(json.dumps(serializable, indent=2))
    print(f"Metrics saved to: {eval_path}\nDone.")


if __name__ == "__main__":
    main(show="--show" in sys.argv)
