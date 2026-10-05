"""
train.py — PHASE 5: MobileNetV2 transfer learning in two stages.
===============================================================
Run locally (CPU smoke test — proves the code runs, NOT for real training):

    .venv\\Scripts\\python src\\train.py --smoke

Real training (Google Colab GPU — see notebooks/waste_classification.ipynb):

    !python train.py --epochs-a 25 --epochs-b 20

The two stages (understand these — they are the heart of transfer learning):

  STAGE A — Feature extraction. The MobileNetV2 base (trained on ImageNet)
  stays FROZEN: its ~2.2M weights never change. Only our new classifier head
  (pooling + dropout + dense) learns. Fast, stable, hard to overfit.

  STAGE B — Fine-tuning. We unfreeze the LAST few base layers (they hold the
  most task-specific patterns) and keep training with a TINY learning rate,
  so the base adapts gently to waste photos without forgetting ImageNet.

Backpropagation refresher: each batch, the loss (how wrong we are) flows
backwards through the network; the optimizer (Adam) nudges every UNFROZEN
weight a small step (learning rate) downhill. Frozen weights are skipped.
Softmax turns the 4 raw outputs into probabilities that sum to 1.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2

# Make `import preprocessing` work whether we run as src/train.py (local)
# or as a flat train.py + preprocessing.py pair (Colab).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import IMG_SIZE, SEED, load_datasets  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT = PROJECT_ROOT / "models"
DEFAULT_HIST = PROJECT_ROOT / "outputs"
CLASS_NAMES_JSON = PROJECT_ROOT / "models" / "class_names.json"
N_CLASSES = 4


def count_per_class(data_dir: Path, split: str, order: list[str]) -> list[int]:
    """Number of train images per class, in the given class order."""
    return [len(list((data_dir / split / c).glob("*.*"))) for c in order]


def build_model(num_classes: int = N_CLASSES,
                dropout: float = 0.3) -> tuple[tf.keras.Model, tf.keras.Model]:
    """MobileNetV2 base (frozen) + new classifier head. Returns (model, base)."""
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False,
                       weights="imagenet")
    base.trainable = False  # STAGE A: freeze everything pretrained

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    # training=False keeps BatchNorm layers in inference mode while frozen.
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # h*w*c -> c (one value per filter)
    x = tf.keras.layers.Dropout(dropout)(x)          # randomly mute neurons (anti-memorize)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def callbacks(best_path: Path, patience_es: int) -> list:
    """EarlyStopping + checkpoint-best + shrink LR when val loss stalls."""
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                         patience=patience_es,
                                         restore_best_weights=True,
                                         verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
                                           mode="max", save_best_only=True,
                                           verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=3, min_lr=1e-7, verbose=1),
    ]


def save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: [float(v) for v in vals]
             for k, vals in history.history.items()}
    path.write_text(json.dumps(clean, indent=2))


def main(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(SEED)
    out_dir, hist_dir = Path(args.models_dir), Path(args.history_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    class_order = json.loads(Path(args.class_names_json).read_text())
    train_ds, val_ds, test_ds, folder_order = load_datasets(
        data_dir, batch_size=args.batch_size, seed=SEED)
    assert folder_order == class_order, "folder order != class_names.json!"
    if args.limit_batches:  # smoke test: only a few batches per epoch
        train_ds = train_ds.take(args.limit_batches)
        val_ds = val_ds.take(args.limit_batches)

    # Class weights: paper has ~2x the images of metal, so its mistakes count
    # less per-image... we invert that: rarer classes get LARGER weights so
    # the loss punishes their mistakes more. Standard imbalance fix.
    counts = count_per_class(data_dir, "train", class_order)
    weights = compute_class_weight("balanced", classes=np.arange(N_CLASSES),
                                   y=np.repeat(np.arange(N_CLASSES), counts))
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print(f"Train counts per class {dict(zip(class_order, counts))}")
    print(f"Class weights          {class_weight}")

    model, base = build_model(dropout=args.dropout)
    model.summary(print_fn=print, line_length=100)
    print(f"Stage A trainable params: {sum(int(np.prod(v.shape)) for v in model.trainable_weights):,}")

    # ---------------- STAGE A: feature extraction -------------------------
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_a),
                  loss="sparse_categorical_crossentropy",  # integer labels 0..3
                  metrics=["accuracy"])
    print(f"\n=== STAGE A: {args.epochs_a} epochs, base frozen, lr={args.lr_a} ===")
    hist_a = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_a,
                       class_weight=class_weight,
                       callbacks=callbacks(out_dir / "stage_a_best.keras",
                                           args.patience))
    save_history(hist_a, hist_dir / "history_stage_a.json")

    # ---------------- STAGE B: fine-tuning --------------------------------
    base.trainable = True
    # Freeze the early layers (generic edges/textures), train only the tail.
    for layer in base.layers[:-args.unfreeze_last]:
        layer.trainable = False
    n_trainable = sum(l.trainable for l in base.layers)
    print(f"\n=== STAGE B: unfroze last {args.unfreeze_last} base layers "
          f"({n_trainable}/{len(base.layers)} base layers trainable), "
          f"lr={args.lr_b} ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_b),  # tiny steps!
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist_b = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_b,
                       class_weight=class_weight,
                       callbacks=callbacks(out_dir / "waste_classifier.keras",
                                           args.patience))
    save_history(hist_b, hist_dir / "history_stage_b.json")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal val accuracy: {val_acc:.4f}  val loss: {val_loss:.4f}")
    print(f"Model -> {out_dir / 'waste_classifier.keras'}")
    print("Test set stays SEALED until Phase 6 (evaluate.py). Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the waste classifier.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--models-dir", default=str(DEFAULT_OUT))
    p.add_argument("--history-dir", default=str(DEFAULT_HIST))
    p.add_argument("--class-names-json", default=str(CLASS_NAMES_JSON))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-a", type=int, default=25)
    p.add_argument("--epochs-b", type=int, default=20)
    p.add_argument("--lr-a", type=float, default=1e-3)
    p.add_argument("--lr-b", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--unfreeze-last", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--limit-batches", type=int, default=0,
                   help=">0 keeps only N batches/epoch (CPU smoke test).")
    p.add_argument("--smoke", action="store_true",
                   help="shortcut: 1 epoch x 3 batches (proves code runs).")
    a = p.parse_args()
    if a.smoke:
        a.epochs_a, a.epochs_b, a.limit_batches = 1, 1, 3
    main(a)
