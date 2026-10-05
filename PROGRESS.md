# Waste Segregation — Progress Log

> Updated by the assistant after every phase. Single source of truth for
> what is done, what is next, and every decision taken.

## Phase status

| # | Phase | Status | Evidence |
|---|-------|--------|----------|
| 1 | Inspection + setup (venv, requirements, structure) | ✅ done | TF 2.20.0 / Keras 3.15.1 verified by import; 2,527 images copied |
| 2 | Dataset exploration (`src/explore_dataset.py`) | ✅ done | Counts printed + `outputs/figures/*.png` generated and viewed |
| 3 | Organic dataset + 4-class split (`src/data_preparation.py`) | ✅ done | `data/processed/` 1,670/356/362; manifest 2,388 rows; 0 cross-split files |
| 4 | Preprocessing + augmentation (`src/preprocessing.py`) | ✅ done | Batches (32,224,224,3) float32 in [-1,1]; folder order == JSON; `outputs/figures/augmentation_preview.png` viewed |
| 5 | MobileNetV2 model (`src/train.py` + Colab notebook) | 🔶 code done + CPU smoke passed; GPU run = user | Smoke: Stage A 5,124 trainable, Stage B 30/154 unfrozen, both `.keras` saved (TEMP); notebook validated, 11 cells |
| 6 | Evaluation (`src/evaluate.py`) | ✅ done | TEST accuracy 0.9282, macro-F1 0.9237 (Stage A was 0.9227/0.9180); curves + CM figures viewed; `outputs/evaluation.json` |
| 7 | Save `models/waste_classifier.keras` + `class_names.json` | ✅ done | Both in `models/` (came back from Colab; verified sizes) |
| 8 | Streamlit `app.py` | ✅ done | Compiles; 1 test image/class all correct (metal 99%, organic 98%, paper 100%, plastic 81%); headless server boot → health 200, no errors |
| 9 | Testing + debugging (headless AppTest: T1-T4) | ✅ done | T1 4/4 correct labels; T2 corrupt→friendly error; T3 missing model→clear error, restored; T4 noise→graceful forced guess, no crash. No code changes needed |
| 10 | README + college report | ✅ done | `README.md` + `docs/project_report.md`; all figures cross-checked vs `evaluation.json`/`split_info.json` |
| — | User fixes OneDrive sync for project folder | ⏳ user | **remind at the very end** |

## Results so far (measured, never invented)

- TrashNet `dataset-resized`: cardboard 403, glass 501, metal 410, paper 594,
  plastic 482, trash 137 = **2,527** (0 unreadable; matches official README).
- 4-class split (seed 42, 70/15/15): paper 997, organic 499, plastic 482,
  metal 410 = **2,388** (train 1,670 / val 356 / test 362).
- Organic source: Mendeley DOI 10.17632/n3gtgm9jxj.3 (CC BY 4.0; cite Nnamoko
  et al. 2022). 9,591 files on disk, first 500 used, 1 internal dupe removed.
- Cross-class byte-duplicate audit: **0 shared files**. Determinism check:
  two runs → identical manifest hash. Leakage check: 0 files in >1 split.
- TEST (sealed, n=362): Stage A 0.9227 acc / 0.9180 macro-F1 → final 0.9282 /
  0.9237. Per-class F1: organic 0.99, paper 0.94, metal 0.90, plastic 0.87.
  Main confusions: metal↔plastic (5+3), paper↔plastic (6+5). Val was 0.9607:
  ~3pp val/test gap = mild selection bias, honestly reported.
- Report figures: `prepared_distribution.png` (NEW, from disk scan of the
  prepared 4-class set, manifest cross-checked) + training curves + confusion
  matrix, all embedded in `docs/project_report.md` (Figures 1–3).
- EXTRA: training notebook Step 5 now verifies the 4 output files exist, then
  zips them into ONE `waste_outputs.zip` for a single download (nbformat-valid,
  all code cells compile). Step 6 added: Google-Drive backup cell, after a
  runtime reconnect wiped a finished training run (Colab disks are ephemeral).

## Decisions log

1. Project lives in `waste-segregation/`; dataset COPIED in (Downloads originals untouched).
2. Venv Python **3.12** (matches Colab; TF 2.20 supported).
3. Training on **Colab GPU** (no local NVIDIA GPU); local venv for explore/eval/app.
4. Paper = paper + cardboard merged. Glass + trash dropped (not organic).
5. `organic/` relocated from `data/dataset-resized/` → `data/organic-raw/`
   (keeps TrashNet copy pristine; user-placed folder, nothing deleted).
6. Paper ≈2× smallest class → handle with class weights in Phase 5, verify with
   per-class metrics in Phase 6.

## Next action

DONE — project complete. Remaining is user-side: the OneDrive sync fix, plus
optional conversion of docs/project_report.md to DOCX for submission.
