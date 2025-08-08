import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import joblib
import os

from src.prepare_sequences import build_sequences_rich
from compare_all import build_baseline_samples

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/aact_extracted.csv"
TRAIN_IDX_PATH = "splits/train_idx.npy"
VAL_IDX_PATH = "splits/val_idx.npy"
OUT_DIR = "results/diagnostics_phase_a"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1. Sequence length distribution
# -----------------------------
print("Loading sequences...")
X, y, lengths, sponsors = build_sequences_rich(DATA_PATH, max_seq_len=10)

seq_lengths = lengths.numpy()
plt.hist(seq_lengths, bins=np.arange(1, 12)-0.5, edgecolor='black')
plt.xlabel("Sequence Length")
plt.ylabel("Number of Samples")
plt.title("Sequence Length Distribution (Sponsor Histories)")
plt.savefig(f"{OUT_DIR}/seq_length_distribution.png")
plt.close()

print(f"Median sequence length: {np.median(seq_lengths):.2f}")
print(f"90th percentile sequence length: {np.percentile(seq_lengths, 90):.2f}")

# -----------------------------
# 2. Failure rate by sequence length
# -----------------------------
fail_rate_by_len = []
for L in np.unique(seq_lengths):
    mask = seq_lengths == L
    fail_rate = y[mask].numpy().mean()
    fail_rate_by_len.append((L, fail_rate))

fail_df = pd.DataFrame(fail_rate_by_len, columns=["seq_len", "fail_rate"])
fail_df.to_csv(f"{OUT_DIR}/fail_rate_by_seq_len.csv", index=False)

plt.plot(fail_df["seq_len"], fail_df["fail_rate"], marker='o')
plt.xlabel("Sequence Length")
plt.ylabel("Failure Rate")
plt.title("Failure Rate vs Sequence Length")
plt.savefig(f"{OUT_DIR}/fail_rate_vs_seq_length.png")
plt.close()

# -----------------------------
# 3. Sponsor size vs. performance (Baseline-3F)
# -----------------------------
print("Evaluating sponsor-size effect...")
tr_idx = np.load(TRAIN_IDX_PATH)
va_idx = np.load(VAL_IDX_PATH)

df = pd.read_csv(DATA_PATH, parse_dates=["start_date"])

hist_len_per_sponsor = (
    df.dropna(subset=["sponsor_name", "start_date"])
      .sort_values(["sponsor_name","start_date"])
      .groupby("sponsor_name").size()
)
hist_len_per_sponsor.describe().to_csv("results/diagnostics_phase_a/true_hist_length_summary.csv")
# quick plot (optional)
import matplotlib.pyplot as plt
plt.figure()
hist_len_per_sponsor.clip(upper=50).hist(bins=50)  # clip to show tail
plt.title("True trials per sponsor (clipped at 50)")
plt.xlabel("# trials for sponsor"); plt.ylabel("count of sponsors")
plt.savefig("results/diagnostics_phase_a/true_hist_length_hist.png"); plt.close()

X3, y3, sponsor_ids = build_baseline_samples(df, max_hist=10, use_categoricals=False)

X3tr, X3va = X3.iloc[tr_idx], X3.iloc[va_idx]
y3tr, y3va = y3[tr_idx], y3[va_idx]
sponsors_va = sponsor_ids[va_idx]

lr = LogisticRegression(max_iter=500, class_weight="balanced")
lr.fit(X3tr, y3tr)
probs = lr.predict_proba(X3va)[:, 1]

# replace the sponsor loop in diagnostics (where you compute per-sponsor AUC)
sponsor_perf = []
for s in np.unique(sponsors_va):
    mask = sponsors_va == s
    if mask.sum() < 5:  # bump minimum samples
        continue
    y_sub = y3va[mask]
    p_sub = probs[mask]
    # require both classes present
    if len(np.unique(y_sub)) < 2:
        continue
    auc = roc_auc_score(y_sub, p_sub)
    sponsor_perf.append((s, mask.sum(), auc))


sponsor_perf_df = pd.DataFrame(sponsor_perf, columns=["sponsor", "n_samples", "auc"])
sponsor_perf_df.to_csv(f"{OUT_DIR}/sponsor_perf_baseline3F.csv", index=False)

# -----------------------------
# 4. SHAP feature importance for Baseline-7F
# -----------------------------
print("Calculating SHAP values for Baseline-7F...")
X7, y7, sponsors7 = build_baseline_samples(df, max_hist=10, use_categoricals=True)

X7tr, X7va = X7.iloc[tr_idx], X7.iloc[va_idx]
y7tr, y7va = y7[tr_idx], y7[va_idx]

lr7 = LogisticRegression(max_iter=500, class_weight="balanced")
lr7.fit(X7tr, y7tr)

explainer = shap.Explainer(lr7, X7tr)  # auto-detects linear model
shap_values = explainer(X7tr)
shap.plots.beeswarm(shap_values, show=False)
plt.title("Baseline-7F SHAP (Train)")
plt.savefig("results/diagnostics_phase_a/shap_baseline7F.png"); plt.close()

from sklearn.inspection import permutation_importance
r = permutation_importance(lr7, X7va, y7va, n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.DataFrame({"feature": X7va.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False)
imp.to_csv("results/diagnostics_phase_a/permutation_importance_baseline7F.csv", index=False)

# -----------------------------
# 5. Sponsor type distribution
# -----------------------------
if "sponsor_type" in df.columns:
    type_counts = df["sponsor_type"].value_counts()
    type_counts.to_csv(f"{OUT_DIR}/sponsor_type_counts.csv")
    type_counts.plot(kind='bar')
    plt.title("Sponsor Type Distribution")
    plt.ylabel("Count")
    plt.savefig(f"{OUT_DIR}/sponsor_type_distribution.png")
    plt.close()

print(f"✅ Diagnostics saved in: {OUT_DIR}")
