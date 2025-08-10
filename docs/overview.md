# 📄 Project Overview — SponsorsRisk

## 1. Problem Statement

Clinical trial sponsors vary widely in their ability to complete studies successfully.
This project develops a **longitudinal sponsor-centric model** to predict the **risk of trial failure** based on a sponsor’s historical trial patterns.
The goal: identify at-risk sponsors early, enabling proactive interventions and improved trial planning.

---

## 2. Data

### 2.1 Source

* **AACT (Aggregate Analysis of ClinicalTrials.gov)** dataset.
* Extracted and preprocessed into `data/aact_extracted.csv` using custom scripts.
* Includes both **structured** trial metadata and temporal ordering of each sponsor’s trial history.

### 2.2 Filters

* Exclude withdrawn trials with no enrollment/start date.
* Limit to trials with valid `start_date`, `phase`, `overall_status`.
* Sponsors must have at least **2 completed trials** to appear in training (ensures longitudinal context).

### 2.3 Label Definition

* Binary classification: **1 = failed**, **0 = not failed**.
* Failure defined as trials with statuses such as `Terminated`, `Suspended`, or `Withdrawn` for non-administrative reasons.
* Labels are assigned per **sponsor** using their latest trial outcome in the sequence.

---

## 3. Features

### 3.1 Numeric trends

* Enrollment counts
* Sequence length
* Year gaps between trials

### 3.2 Categorical encodings

* Phase (`Phase 1`, `Phase 2`, …)
* Allocation (`Randomized`, `Non-Randomized`)
* Masking (`None`, `Single`, `Double`, …)
* Primary purpose
* Intervention type
* Overall status

### 3.3 Temporal structure

* Trials sorted by `start_date` and padded/truncated to `max_seq_len` (default: 10).

---

## 4. Models

### 4.1 GRU baseline

* Sequence of numeric features only.

### 4.2 Transformer

* Sequence of numeric features only, self-attention for long-range dependencies.

### 4.3 Combined model

* Numeric features processed via GRU
* Categorical features embedded and fused with numeric sequence

---

## 5. Training & Evaluation

### 5.1 Split strategy

* **Sponsor-level split**: no sponsor appears in both train and validation.
* Default: 80/20 train/validation.

### 5.2 Loss

* Weighted `BCEWithLogitsLoss` with `pos_weight` to counter class imbalance.

### 5.3 Metrics

* **ROC AUC** (`val_auc`)
* **PR AUC** (`val_prauc`)
* **Best-F1 threshold** (`val_best_thr`)
* **Best-F1 score** (`val_best_f1`)

---

## 6. Calibration

* Isotonic regression applied to training predictions.
* Calibrated probabilities improve decision threshold reliability without changing ranking metrics (AUC).

---

## 7. Key Results

From `compare_all.py` (default 30 epochs, batch 256):

| Model              | Val AUC | Val PR-AUC | Best-F1 | Threshold |
| ------------------ | ------: | ---------: | ------: | --------: |
| Combined-9+4       |   0.666 |      0.315 |   0.384 |     0.536 |
| Transformer-9ch    |   0.660 |      0.314 |   0.376 |     0.584 |
| GRU-9ch            |   0.659 |      0.309 |   0.379 |     0.565 |
| Baseline-7F+trends |   0.654 |      0.308 |   0.374 |     0.485 |
| Baseline-3F+trends |   0.647 |      0.303 |   0.375 |     0.482 |

Calibrated curves and slice analyses (by sponsor type and history length) are in `results/plots/`.

---

## 8. Interpretability

* Reliability curves (uncalibrated vs isotonic) for calibration assessment.
* Slice-level AUC by **sponsor type** and **history length** for fairness/bias analysis.
* Future: SHAP-based feature importance for sponsor risk factors.

---

## 9. Ethics & Responsible Use

* Predictions should **not** be used to exclude sponsors from funding or trial participation without human review.
* The model reflects patterns in historical data — potential biases (e.g., geography, sponsor size) must be monitored.
* Calibration improves probability reliability, but predictions remain probabilistic.

---

## 10. Reproducibility

To reproduce:

```bash
# 1. Save splits
python scripts/save_splits.py

# 2. Train a model
python run_train_combined.py

# 3. Compare models & calibrate
python compare_all.py --epochs 30 --batch 256 --device cuda

# 4. Generate real examples for API demo
python -m scripts.make_real_examples_from_val

# 5. Serve inference API
python -m uvicorn scripts.serve_inference:app --reload
```

All key artifacts (models, calibrators, thresholds, plots) are versioned in `models/` and `results/`.
