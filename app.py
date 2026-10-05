"""
app.py — PHASE 8: Streamlit web application.
============================================
Run from the project root:

    .venv\\Scripts\\streamlit run app.py

Upload a JPG/JPEG/PNG waste photo → preview → Predict → category +
confidence + all-class scores. Preprocessing is imported from
src/preprocessing.py, so inference can never drift from training.
"""

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing import preprocess_pil_image  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "waste_classifier.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

# Below this confidence we warn the user. Honest note: this does NOT reliably
# detect non-waste photos — a 4-class model always picks one of its 4 classes.
LOW_CONFIDENCE_THRESHOLD = 0.60


@st.cache_resource
def load_model_and_labels():
    """Load once per server start (not once per click)."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Train it first (notebooks/waste_classification.ipynb on Colab), "
            "then copy waste_classifier.keras into models/."
        )
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class map not found: {CLASS_NAMES_PATH}")
    import tensorflow as tf  # local import: faster `streamlit --help`, etc.
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text())
    return model, class_names


def predict(model, class_names, img: Image.Image):
    """RGB -> 224x224 -> MobileNetV2 scaling -> predict -> (label, conf, all)."""
    batch = preprocess_pil_image(img)  # shape (1, 224, 224, 3), in [-1, 1]
    probs = [float(p) for p in model.predict(batch, verbose=0)[0]]
    best = int(max(range(len(probs)), key=probs.__getitem__))
    return class_names[best], probs[best], probs


st.set_page_config(page_title="Waste Segregation", layout="centered")
st.title("Waste Segregation Using Image Classification")
st.write(
    "Upload a photo of a waste item and the model predicts **plastic**, "
    "**paper**, **metal**, or **organic**. "
    "MobileNetV2 + transfer learning, trained on TrashNet + organic photos. "
    "Sealed-test accuracy: **92.8%** (macro-F1 0.924)."
)
st.caption(
    "Scores are the model's softmax confidence — a useful estimate, not a "
    "guaranteed true probability."
)

try:
    model, class_names = load_model_and_labels()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Choose a waste photo",
                            type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Waiting for an image… (JPG, JPEG, or PNG)")
    st.stop()

try:
    Image.open(uploaded).verify()  # prove it decodes before trusting it
    uploaded.seek(0)
    img = Image.open(uploaded)
    img.load()
except Exception:
    st.error("That file is not a readable image. Please upload a valid "
             "JPG, JPEG, or PNG photo.")
    st.stop()

left, right = st.columns(2)
with left:
    st.image(img, caption="Uploaded image", use_container_width=True)
with right:
    if st.button("Predict", type="primary"):
        with st.spinner("Classifying…"):
            label, confidence, probs = predict(model, class_names, img)
        st.success(f"**{label.capitalize()}** — {confidence:.1%} confidence")
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"Low confidence (under {LOW_CONFIDENCE_THRESHOLD:.0%}). "
                "The photo may be blurry, show multiple objects, or not be "
                "waste at all — this warning alone cannot detect non-waste "
                "images, so treat the label with caution."
            )
        st.subheader("All categories")
        for name, p in sorted(zip(class_names, probs),
                             key=lambda t: t[1], reverse=True):
            st.progress(p, text=f"{name.capitalize()}: {p:.1%}")
    else:
        st.write("Press **Predict** to classify the photo.")
