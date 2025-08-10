# scripts/build_paper_pack.py
import os, argparse, json, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# ---------- Defaults / paths ----------
RESULTS_DIR = "results"
PROB_DIR    = os.path.join(RESULTS_DIR, "probs")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

MODEL_KEYS = {
    "baseline3": "Baseline-3F+trends",
    "baseline7": "Baseline-7F+trends",
    "gru9":      "GRU-9ch",
    "tx9":       "Transformer-9ch",
    "comb9p4":   "Combined-9+4",
}
# Consistent palette (print-friendly)
PALETTE = {
    "Baseline-3F+trends":   "#1f77b4",
    "Baseline-7F+trends":   "#ff7f0e",
    "GRU-9ch":              "#2ca02c",
    "Transformer-9ch":      "#d62728",
    "Combined-9+4":         "#9467bd",
}

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# ---------- Load saved probabilities ----------
def load_probs(split="val"):
    """Return dict: model_display_name -> (y, p) for given split."""
    out = {}
    for key, name in MODEL_KEYS.items():
        p_path = os.path.join(PROB_DIR, f"{key}_{split}_probs.npy")
        y_path = os.path.join(PROB_DIR, f"{key}_{split}_labels.npy")
        if os.path.exists(p_path) and os.path.exists(y_path):
            out[name] = (np.load(y_path), np.load(p_path))
    return out

# ---------- Plot helpers (consistent style) ----------
def set_style():
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

def save_fig(path, dpi=300):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_roc_overlay(models, split, outdir, dpi):
    set_style()
    plt.figure()
    for name, (y, p) in models.items():
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=PALETTE.get(name))
    plt.plot([0,1], [0,1], linestyle="--", color="#8c564b")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({split})"); plt.legend(loc="lower right")
    out = os.path.join(outdir, f"roc_{split}_overlay.png")
    save_fig(out, dpi)
    return out

def plot_pr_overlay(models, split, outdir, dpi):
    set_style()
    plt.figure()
    for name, (y, p) in models.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=PALETTE.get(name))
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR ({split})"); plt.legend(loc="lower left")
    out = os.path.join(outdir, f"pr_{split}_overlay.png")
    save_fig(out, dpi)
    return out

def plot_reliability_combined(split, outdir, dpi):
    """Fig 3: Combined-9+4 uncalibrated vs isotonic-calibrated."""
    set_style()
    key = "comb9p4"
    p_tr = np.load(os.path.join(PROB_DIR, f"{key}_train_probs.npy"))
    y_tr = np.load(os.path.join(PROB_DIR, f"{key}_train_labels.npy"))
    p_va = np.load(os.path.join(PROB_DIR, f"{key}_{split}_probs.npy"))
    y_va = np.load(os.path.join(PROB_DIR, f"{key}_{split}_labels.npy"))

    prob_true_u, prob_pred_u = calibration_curve(y_va, p_va, n_bins=10, strategy="quantile")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    p_va_iso = iso.transform(p_va)
    prob_true_c, prob_pred_c = calibration_curve(y_va, p_va_iso, n_bins=10, strategy="quantile")

    plt.figure()
    plt.plot([0,1], [0,1], linestyle="--", color="#1f1f1f")
    plt.plot(prob_pred_u, prob_true_u, marker="o", label="Uncalibrated", color="#ff7f0e")
    plt.plot(prob_pred_c, prob_true_c, marker="o", label="Isotonic-calibrated", color="#2ca02c")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Reliability (Combined-9+4, {split})"); plt.legend(loc="upper left")
    out = os.path.join(outdir, f"reliability_{split}_combined_uncal_vs_iso.png")
    save_fig(out, dpi)
    return out

def plot_slices_sponsor_type(model_display, outdir, dpi):
    """Fig 4: AUC by sponsor type (calibrated)."""
    path = os.path.join(RESULTS_DIR, "slices_sponsor_type.csv")
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path)
    dfm = df[df["model"] == model_display].copy()
    if dfm.empty: return ""
    dfm = dfm.sort_values("auc", ascending=False)
    set_style()
    plt.figure()
    plt.barh(dfm["sponsor_type"], dfm["auc"], color=PALETTE.get(model_display))
    for i, (auc, n) in enumerate(zip(dfm["auc"], dfm["n"])):
        plt.text(auc + 0.003, i, f"n={n}", va="center", fontsize=10)
    plt.xlabel("AUC"); plt.ylabel("Sponsor type")
    plt.title(f"AUC by sponsor type ({model_display}, calibrated)")
    out = os.path.join(outdir, f"slices_sponsor_type_{model_display.replace(' ','_')}.png")
    save_fig(out, dpi)
    return out

def plot_slices_histlen(model_display, outdir, dpi):
    """Fig 5: AUC by history length (calibrated)."""
    path = os.path.join(RESULTS_DIR, "slices_histlen.csv")
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path)
    dfm = df[df["model"] == model_display].copy()
    if dfm.empty: return ""
    order = ["1-2","3-5","6-10"]
    dfm["bucket"] = pd.Categorical(dfm["bucket"], categories=order, ordered=True)
    dfm = dfm.sort_values("bucket")
    set_style()
    plt.figure()
    plt.bar(dfm["bucket"], dfm["auc"], color=PALETTE.get(model_display))
    for x, (auc, n) in zip(dfm["bucket"], zip(dfm["auc"], dfm["n"])):
        plt.text(x, auc + 0.003, f"n={n}", ha="center", fontsize=10)
    plt.xlabel("History length (trials)"); plt.ylabel("AUC")
    plt.title(f"AUC by history length ({model_display}, calibrated)")
    out = os.path.join(outdir, f"slices_histlen_{model_display.replace(' ','_')}.png")
    save_fig(out, dpi)
    return out

def copy_metrics(dst_dir):
    for fn in ["metrics_overall.csv", "metrics_overall.json",
               "metrics_overall_calibrated.csv", "metrics_overall_calibrated.json"]:
        src = os.path.join(RESULTS_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))

def write_readme(dst_dir, commit, figs):
    lines = [
        "# SponsorsRisk – Paper Pack",
        "",
        f"- Commit: `{commit}`",
        "- This folder contains the exact tables and figures used in the paper.",
        "",
        "## Figures",
    ]
    for tag, path in figs.items():
        lines.append(f"- {tag}: `{os.path.basename(path)}`")
    lines += [
        "",
        "## Tables",
        "- `metrics_overall.csv` / `metrics_overall_calibrated.csv` (AUC, PR-AUC, best-F1 & threshold)",
        "- JSON mirrors of the above for programmatic use.",
        "",
        "Re-generate with:",
        "```bash",
        "python scripts/build_paper_pack.py --split val --dpi 300 --out docs/paper_pack",
        "```",
    ]
    with open(os.path.join(dst_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val"], default="val")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="docs/paper_pack")
    args = ap.parse_args()

    ensure_dir(PLOTS_DIR)       # where plots also live
    ensure_dir(args.out)        # paper pack target

    # 1) build overlay ROC/PR (Fig 1 & 2)
    models = load_probs(args.split)
    if not models:
        raise SystemExit(f"No saved probabilities in {PROB_DIR}. Run compare_all.py first.")
    fig1 = plot_roc_overlay(models, args.split, PLOTS_DIR, args.dpi)
    fig2 = plot_pr_overlay(models, args.split, PLOTS_DIR, args.dpi)

    # 2) reliability (Fig 3)
    fig3 = plot_reliability_combined(args.split, PLOTS_DIR, args.dpi)

    # 3) slice bars (Fig 4 & 5) — calibrated Combined-9+4
    fig4 = plot_slices_sponsor_type("Combined-9+4", PLOTS_DIR, args.dpi)
    fig5 = plot_slices_histlen("Combined-9+4", PLOTS_DIR, args.dpi)

    # 4) copy to paper pack + metrics + README with git hash
    commit = git_commit()
    figs = {
        "Figure 1 (ROC overlay)": fig1,
        "Figure 2 (PR overlay)": fig2,
        "Figure 3 (Reliability, uncal vs iso, Combined-9+4)": fig3,
        "Figure 4 (AUC by sponsor type, Combined-9+4)": fig4,
        "Figure 5 (AUC by history length, Combined-9+4)": fig5,
    }
    for _, src in figs.items():
        if src and os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, os.path.basename(src)))
    copy_metrics(args.out)
    write_readme(args.out, commit, figs)

    print("✅ Paper pack built at", args.out)
    for tag, p in figs.items():
        print(f" - {tag}: {os.path.join(args.out, os.path.basename(p))}")

if __name__ == "__main__":
    main()
