# scripts/plot_curves.py
import os, argparse, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

OUTDIR = "results/plots"
PROBDIR = "results/probs"
os.makedirs(OUTDIR, exist_ok=True)

MODEL_KEYS = {
    "baseline3": "Baseline-3F+trends",
    "baseline7": "Baseline-7F+trends",
    "gru9":      "GRU-9ch",
    "tx9":       "Transformer-9ch",
    "comb9p4":   "Combined-9+4",
}

def _load(split_key: str):
    """Return dict: key -> (y, p) for a given split ('train' or 'val')."""
    out = {}
    for k in MODEL_KEYS:
        p_path = os.path.join(PROBDIR, f"{k}_{split_key}_probs.npy")
        y_path = os.path.join(PROBDIR, f"{k}_{split_key}_labels.npy")
        if os.path.exists(p_path) and os.path.exists(y_path):
            p = np.load(p_path)
            y = np.load(y_path)
            out[MODEL_KEYS[k]] = (y, p)
    return out

def _plot_save(path, fig):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_roc(models, split="val"):
    paths = {}
    for name, (y, p) in models.items():
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC ({split})")
        plt.legend(loc="lower right")
        out = os.path.join(OUTDIR, f"roc_{split}_{name.replace(' ','_')}.png")
        _plot_save(out, fig)
        paths[name] = out
    return paths

def plot_pr(models, split="val"):
    paths = {}
    for name, (y, p) in models.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        fig = plt.figure()
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({split})")
        plt.legend(loc="lower left")
        out = os.path.join(OUTDIR, f"pr_{split}_{name.replace(' ','_')}.png")
        _plot_save(out, fig)
        paths[name] = out
    return paths

def plot_reliability(models, split="val", n_bins=10):
    paths = {}
    for name, (y, p) in models.items():
        prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        fig = plt.figure()
        plt.plot([0,1],[0,1], linestyle="--")
        plt.plot(prob_pred, prob_true, marker="o", label=name)
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.title(f"Reliability ({split})")
        plt.legend(loc="upper left")
        out = os.path.join(OUTDIR, f"reliability_{split}_{name.replace(' ','_')}.png")
        _plot_save(out, fig)
        paths[name] = out
    return paths

def plot_roc_overlay(models, split="val"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fig = plt.figure()
    for name, (y, p) in models.items():
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({split})"); plt.legend(loc="lower right")
    out = os.path.join(OUTDIR, f"roc_{split}_overlay.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out

def plot_pr_overlay(models, split="val"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig = plt.figure()
    for name, (y, p) in models.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR ({split})"); plt.legend(loc="lower left")
    out = os.path.join(OUTDIR, f"pr_{split}_overlay.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out

# --- Fig 3: reliability overlay for Combined-9+4 (uncal vs iso) ---
def plot_reliability_combined_calibrated(split="val"):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    import numpy as np, os

    PROBDIR = "results/probs"
    OUTDIR  = "results/plots"
    os.makedirs(OUTDIR, exist_ok=True)

    # load train/val for Combined-9+4
    p_tr = np.load(os.path.join(PROBDIR, "comb9p4_train_probs.npy"))
    y_tr = np.load(os.path.join(PROBDIR, "comb9p4_train_labels.npy"))
    p_va = np.load(os.path.join(PROBDIR, f"comb9p4_{split}_probs.npy"))
    y_va = np.load(os.path.join(PROBDIR, f"comb9p4_{split}_labels.npy"))

    # uncalibrated reliability
    prob_true_u, prob_pred_u = calibration_curve(y_va, p_va, n_bins=10, strategy="quantile")

    # isotonic on train → transform val
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    p_va_iso = iso.transform(p_va)
    prob_true_c, prob_pred_c = calibration_curve(y_va, p_va_iso, n_bins=10, strategy="quantile")

    fig = plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(prob_pred_u, prob_true_u, marker="o", label="Uncalibrated")
    plt.plot(prob_pred_c, prob_true_c, marker="o", label="Isotonic-calibrated")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Reliability (Combined-9+4, {split})")
    plt.legend(loc="upper left")
    out = os.path.join(OUTDIR, f"reliability_{split}_combined_uncal_vs_iso.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("✅ Fig 3 saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val","both"], default="val")
    args = ap.parse_args()

    splits = ["val"] if args.split != "both" else ["train","val"]
    rows = []
    for split in splits:
        models = _load(split)
        overlay_roc = plot_roc_overlay(models, split)
        overlay_pr  = plot_pr_overlay(models, split)

        if not models:
            print(f"[plot_curves] No probs found for split={split} under {PROBDIR}. Run compare_all.py first.")
            continue
        roc_paths = plot_roc(models, split)
        pr_paths = plot_pr(models, split)
        rel_paths = plot_reliability(models, split)
        for name in models:
            rows.append({
                "split": split,
                "model": name,
                "roc_path": roc_paths.get(name, ""),
                "pr_path": pr_paths.get(name, ""),
                "reliability_path": rel_paths.get(name, ""),
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTDIR, "plot_index.csv"), index=False)
        print("✅ Plots saved to", OUTDIR)
    else:
        print("No plots generated.")

    # Fig 3
    plot_reliability_combined_calibrated(split="val")

if __name__ == "__main__":
    main()
