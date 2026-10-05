# Waste Segregation Using Image Classification

Computer-vision system that accepts a photo of a waste item and predicts its
category — **plastic, paper, metal, or organic** — with a confidence score,
served through a Streamlit web app.

**Headline result (sealed test set, n = 362): accuracy 92.8%, macro-F1 0.924.**
Stage-A-only baseline was 92.3% / 0.918, so fine-tuning added +0.5pp.

## Problem statement

Mixed household waste makes recycling expensive: sorting is manual, slow, and
error-prone. An image classifier at the bin (or in a phone app) can route each
item to the right stream — plastic, paper, metal, or organic — cutting
contamination and landfill load.

## Objectives

1. Build a 4-class waste image classifier with transfer learning.
2. Reach honest, reproducible test metrics (no validation-peeking).
3. Ship a usable web demo: upload → category + confidence.
4. Document every step so the full ML workflow is learnable and repeatable.

## Dataset sources and class mapping

| Our class | Source | Images |
|---|---|---:|
| Plastic | TrashNet `plastic/` | 482 |
| Paper | TrashNet `paper/` + `cardboard/` (same fibre family) | 997 |
| Metal | TrashNet `metal/` | 410 |
| Organic | Mendeley DOI 10.17632/n3gtgm9jxj.3, first 500 sorted (1 dupe removed → 499) | 499 |
| — | TrashNet `glass/` + `trash/` **excluded** (glass is not organic; trash is mixed) | — |
| **Total** | | **2,388** |

- TrashNet (Thung & Yang, Stanford CS229 2016): 2,527 verified images, 0 unreadable.
  Repo licence MIT; dataset requires citation (see References).
- Organic set (Nnamoko et al. 2022): 13,880 organic photos, **CC BY 4.0** —
  free to use with attribution (see References). Our copy holds 9,591 files.
- Split: stratified **70/15/15** (train 1,670 / val 356 / test 362), seed 42.
  Exact-duplicate files removed by MD5; audit found 0 files shared across
  classes and 0 files in more than one split. Manifest: `data/splits/`.

## Technologies

Python 3.12 · TensorFlow 2.20 + Keras 3 (MobileNetV2, ImageNet weights) ·
NumPy · Pillow · Matplotlib · scikit-learn · Streamlit · Google Colab (T4 GPU
for training; laptop has no NVIDIA GPU so local runs are CPU-only).

## Installation (Windows, VS Code)

```powershell
cd waste-segregation
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## How to train the model

Fast smoke test (CPU, ~5 min, proves the code runs — not real training):

```powershell
.venv\Scripts\python src\train.py --smoke
```

Real training (Colab GPU, ~10–25 min):

```powershell
Compress-Archive -Path data\processed -DestinationPath processed_colab.zip -Force
```

Upload `processed_colab.zip`, `src/train.py`, `src/preprocessing.py`,
`models/class_names.json` into `notebooks/waste_classification.ipynb` on
Colab (T4 GPU), run all cells, download `waste_classifier.keras`,
`stage_a_best.keras`, and both history JSONs back into `models/` and `outputs/`.
Two stages: **A** — frozen base, Adam 1e-3, 25 epochs; **B** — last 30 base
layers unfrozen, Adam 1e-5, 20 epochs. EarlyStopping (restore best) +
ModelCheckpoint + ReduceLROnPlateau throughout; class weights compensate the
~2× paper majority.

## How to evaluate

```powershell
.venv\Scripts\python src\evaluate.py
```

Scores both models on the sealed test set; writes `outputs/evaluation.json`,
`outputs/figures/training_curves.png`, `outputs/figures/confusion_matrix.png`.

## How to run the app

```powershell
.venv\Scripts\streamlit run app.py
```

Upload JPG/JPEG/PNG → preview → **Predict** → category, confidence, and
per-class bars. A sub-60% score raises a caution flag (it is *not* a reliable
non-waste detector — the model always picks one of its 4 classes).

## Results (actual, test set n = 362)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| metal | 0.89 | 0.90 | 0.90 |
| organic | 0.99 | 0.99 | 0.99 |
| paper | 0.95 | 0.93 | 0.94 |
| plastic | 0.86 | 0.89 | 0.87 |

Accuracy **0.928**, macro-F1 **0.924**. Main confusions are material-adjacent:
metal↔plastic (shiny cans vs bottles), paper↔plastic (wrappers/films).
Validation accuracy was 0.961 — the ~3pp val/test gap is expected selection
bias (checkpoints are chosen on val); **0.928 is the unbiased figure**.

## Reproduce the pipeline

```powershell
.venv\Scripts\python src\explore_dataset.py      # counts + figures
.venv\Scripts\python src\data_preparation.py --organic-src data\organic-raw
.venv\Scripts\python src\preprocessing.py        # batch/range/augment check
.venv\Scripts\python src\evaluate.py             # test metrics + figures
```

## Limitations and future scope

- Clean white-background training photos vs messy real-world bins (domain gap);
  augmentation helps, real-bin photos would help more.
- No reliable out-of-distribution detection (a toy photo still gets a label).
- Paper ≈2× the smallest class; class weights contain it but more metal/plastic
  photos would be better.
- Mild source label noise in the organic set.
- Next: real-bin photo collection, OOD/reject option, lighter model export
  (TFLite) for on-device sorting, top-2 display when scores are close.

## Project structure

```
waste-segregation/
  app.py  requirements.txt  PROGRESS.md  README.md
  src/         explore_dataset.py  data_preparation.py
               preprocessing.py  train.py  evaluate.py
  notebooks/  waste_classification.ipynb   (Colab GPU training)
  models/      waste_classifier.keras  stage_a_best.keras  class_names.json
  data/        dataset-resized/  organic-raw/  processed/  splits/
  outputs/     figures/  evaluation.json  history_stage_*.json
  docs/        project_report.md
```

## References

- Thung & Yang (2016), TrashNet. GitHub `garythung/trashnet` (MIT code);
  dataset: HuggingFace `garythung/trashnet`. Please cite the repository.
- Nnamoko, Barrowclough & Procter (2022), "Solid Waste Image Classification
  Using Deep CNN", *Infrastructures* 7(4):47. doi:10.3390/infrastructures7040047.
  Dataset (CC BY 4.0): doi:10.17632/n3gtgm9jxj.3 (via Kaggle
  `techsash/waste-classification-data`).
- Howard et al. (2018), "MobileNetV2: Inverted Residuals and Linear
  Bottlenecks", arXiv:1801.04381.
