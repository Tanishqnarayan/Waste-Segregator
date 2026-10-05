"""
preprocessing.py — PHASE 4: Preprocessing + augmentation.
==========================================================
Run from the project root to verify the pipeline:

    .venv\\Scripts\\python src\\preprocessing.py

The two ideas in this file (read these — they are exam-worthy):

1. NORMALIZATION — MobileNetV2 was trained on ImageNet photos scaled to
   [-1, +1]. If we feed it [0, 255] pixels instead, the numbers are ~100x
   bigger than what its weights expect, so predictions become garbage.
   `mobilenet_v2.preprocess_input` does exactly this scaling: x/127.5 - 1.
   We apply the SAME function during training AND in app.py at inference —
   this module is the single shared definition, so they can never drift apart.

2. AUGMENTATION — we only have 1,670 training photos. Showing the model
   slightly altered copies (flipped / rotated / zoomed) each epoch teaches
   it the *object*, not the exact pixels: a bottle is a bottle even mirrored.
   Augmentation runs ONLY on training data — validation/test stay untouched
   so our scores measure honest performance on real photos.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Settings (tune these; everything else follows) ------------------------
IMG_SIZE = (224, 224)   # MobileNetV2's native input size. Smaller = faster
BATCH_SIZE = 32         # images per training step. 32 fits easily on CPU/GPU
SEED = 42               # shuffling seed — same seed, same batches every run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def build_augmentation(seed: int = SEED) -> tf.keras.Sequential:
    """Train-only random transforms. Inactive at inference (training=False)."""
    return tf.keras.Sequential(
        [
            # Mirror left-right: a can is still a can when flipped.
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            # Small tilts (±10% of 2π ≈ ±36°... actually factor 0.1 = ±10% of
            # half-circle each way): photos are rarely perfectly straight.
            tf.keras.layers.RandomRotation(0.1, seed=seed),
            # Zoom in/out by up to 20%: object may fill more or less frame.
            tf.keras.layers.RandomZoom(0.2, seed=seed),
        ],
        name="augmentation",
    )


def load_datasets(processed_dir: Path = PROCESSED_DIR,
                  batch_size: int = BATCH_SIZE,
                  seed: int = SEED,
                  augment: bool = True
                  ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset,
                             list[str]]:
    """Load train/val/test as tf.data pipelines, preprocessed for MobileNetV2.

    Steps per image: decode -> RGB resize 224x224 (done by the loader) ->
    float32 -> [train only] augment -> preprocess_input ([-1, 1]).

    Returns (train_ds, val_ds, test_ds, class_names). Class names are
    returned separately because .prefetch() datasets no longer carry them.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        processed_dir / "train", image_size=IMG_SIZE, batch_size=batch_size,
        shuffle=True, seed=seed)
    # Grab the label order NOW: .map()/.prefetch() wrappers drop it.
    class_names: list[str] = list(train_ds.class_names)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        processed_dir / "val", image_size=IMG_SIZE, batch_size=batch_size,
        shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        processed_dir / "test", image_size=IMG_SIZE, batch_size=batch_size,
        shuffle=False)

    augmentation = build_augmentation(seed) if augment else None

    def prepare(x, y, training: bool):
        x = tf.cast(x, tf.float32)
        if training and augmentation is not None:
            x = augmentation(x, training=True)
        return preprocess_input(x), y  # [0,255] -> [-1,1], THE MobileNetV2 way

    train_ds = train_ds.map(lambda x, y: prepare(x, y, True),
                            num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: prepare(x, y, False),
                        num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: prepare(x, y, False),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # prefetch overlaps data loading with training (free speed, less waiting).
    return (train_ds.prefetch(tf.data.AUTOTUNE),
            val_ds.prefetch(tf.data.AUTOTUNE),
            test_ds.prefetch(tf.data.AUTOTUNE),
            class_names)


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """Preprocess ONE image exactly like training. Used by app.py (Phase 8).

    RGB -> resize 224x224 -> MobileNetV2 scaling -> add batch dimension.
    Returns float32 array of shape (1, 224, 224, 3), values in [-1, 1].
    """
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = preprocess_input(np.asarray(img, dtype=np.float32))
    return np.expand_dims(arr, axis=0)


def save_augmentation_preview(train_ds: tf.data.Dataset,
                              rows: int = 3, cols: int = 4) -> Path:
    """Save a grid showing what augmentation does to real training images."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "augmentation_preview.png"

    augmentation = build_augmentation(SEED)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for i in range(rows):
        # Grab one raw batch, take image i, show `cols` random variants.
        for x_raw, _ in tf.keras.utils.image_dataset_from_directory(
                PROCESSED_DIR / "train", image_size=IMG_SIZE,
                batch_size=BATCH_SIZE, shuffle=True, seed=SEED + i).take(1):
            base = tf.cast(x_raw[i], tf.float32)
            for j in range(cols):
                variant = augmentation(tf.expand_dims(base, 0), training=True)[0]
                shown = (variant.numpy() / 255.0).clip(0, 1)  # back to viewable
                axes[i, j].imshow(shown)
                axes[i, j].set_title(f"img {i + 1}, variant {j + 1}", fontsize=9)
                axes[i, j].axis("off")
    fig.suptitle("Same training photo + random flip/rotation/zoom", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(show: bool = False) -> None:
    if not show:
        matplotlib.use("Agg")
    train_ds, val_ds, test_ds, folder_order = load_datasets()

    print(f"Class order from folders : {folder_order}")
    saved = __import__("json").loads(
        (PROJECT_ROOT / "models" / "class_names.json").read_text())
    print(f"Class order in JSON      : {saved}")
    print("MATCH [ok]" if list(folder_order) == saved
          else "MISMATCH [fix before training!]")

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        x, y = next(iter(ds))
        print(f"{name:<6} batch {tuple(x.shape)}  dtype={x.dtype.name}  "
              f"pixel range [{tf.reduce_min(x).numpy():+.2f}, "
              f"{tf.reduce_max(x).numpy():+.2f}]  "
              f"batches={len(ds)}")
    # Range must sit inside [-1, 1]: proof preprocess_input ran.
    probe = next(iter(train_ds))[0]
    lo, hi = tf.reduce_min(probe).numpy(), tf.reduce_max(probe).numpy()
    assert -1.0 <= lo and hi <= 1.0, f"pixels out of [-1,1]: [{lo}, {hi}]"
    print(f"Normalization check passed: train pixels in [{lo:+.2f}, {hi:+.2f}]")

    preview = save_augmentation_preview(train_ds)
    print(f"Augmentation preview saved to: {preview}")
    print("Done.")


if __name__ == "__main__":
    import sys

    main(show="--show" in sys.argv)
