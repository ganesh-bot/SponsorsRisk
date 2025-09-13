# scripts/plot_curves.py
import os, argparse, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

FMT = "png"  # CLI will override in main()
OUTDIR = "results/plots_smartgencon"
PROBDIR = "results/probs_smartgencon"
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
    fig.savefig(path, dpi=plt.rcParams.get("figure.dpi", 160), bbox_inches="tight")
    plt.close(fig)

def _safe_path(name, split, suffix, fmt=None):
    if fmt is None:
        fmt = FMT
    return os.path.join(OUTDIR, f"{suffix}_{split}_{name.replace(' ','_')}.{fmt}")

def _ece_brier(y, p, n_bins=10):
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    bins = np.digitize(p, np.quantile(p, np.linspace(0,1,n_bins+1)[1:-1]))
    w = np.bincount(bins, minlength=n_bins)
    w = w / max(w.sum(), 1)
    ece = np.sum(w * np.abs(prob_true - prob_pred))
    brier = np.mean((p - y)**2)
    return float(ece), float(brier)

def _fit_iso(train_y, train_p):
    iso = IsotonicRegression(out_of_bounds="clip")
    return iso.fit(train_p, train_y)

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
        # out = os.path.join(OUTDIR, f"roc_{split}_{name.replace(' ','_')}.png")
        out = _safe_path(name, split, "roc")
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
        # out = os.path.join(OUTDIR, f"pr_{split}_{name.replace(' ','_')}.png")
        out = _safe_path(name, split, "pr")
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
        # out = os.path.join(OUTDIR, f"reliability_{split}_{name.replace(' ','_')}.png")
        out = _safe_path(name, split, "reliability")
        _plot_save(out, fig)
        paths[name] = out
    return paths

def plot_roc_overlay(models, split="val"):
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
def plot_reliability_calibrated_for_all(models, split="val", n_bins=10):
    saved = []
    for key, pretty in MODEL_KEYS.items():
        if pretty not in models:
            continue
        p_tr_path = os.path.join(PROBDIR, f"{key}_train_probs.npy")
        y_tr_path = os.path.join(PROBDIR, f"{key}_train_labels.npy")
        if not (os.path.exists(p_tr_path) and os.path.exists(y_tr_path)):
            continue
        y_va, p_va = models[pretty]
        p_tr, y_tr = np.load(p_tr_path), np.load(y_tr_path)
        iso = _fit_iso(y_tr, p_tr)
        p_va_iso = iso.transform(p_va)
        prob_true_u, prob_pred_u = calibration_curve(y_va, p_va, n_bins=n_bins, strategy="quantile")
        prob_true_c, prob_pred_c = calibration_curve(y_va, p_va_iso, n_bins=n_bins, strategy="quantile")
        fig = plt.figure()
        plt.plot([0,1],[0,1],"--")
        plt.plot(prob_pred_u, prob_true_u, marker="o", label=f"{MODEL_KEYS[key]} – Uncal")
        plt.plot(prob_pred_c, prob_true_c, marker="o", label=f"{MODEL_KEYS[key]} – Isotonic")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.title(f"Reliability (calibration overlay, {split})")
        plt.legend(loc="upper left")
        out = _safe_path(MODEL_KEYS[key], split, "reliability_uncal_vs_iso")
        _plot_save(out, fig)
        saved.append(out)
    if saved:
        print("✅ Calibrated reliability saved:", len(saved))
    return saved

def _maybe_groups(split):
    path = os.path.join(PROBDIR, f"groups_{split}.npy")
    return np.load(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val","both"], default="val")
    ap.add_argument("--format", choices=["png","svg"], default="png")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--with_calibration", action="store_true", help="Plot uncal vs isotonic overlays for all models.")
    ap.add_argument("--with_groupwise", action="store_true", help="Compute group-wise ECE/Brier if groups_{split}.npy exists.")

    args = ap.parse_args()
    plt.rcParams["figure.dpi"] = args.dpi
    plt.rcParams["savefig.format"] = args.format

    splits = ["val"] if args.split != "both" else ["train","val"]
    global FMT
    FMT = args.format
    rows = []
    for split in splits:
        models = _load(split)
        if not models:
            print(f"[plot_curves] No probs found for split={split} under {PROBDIR}. Run compare_all.py first.")
            continue
        overlay_roc = plot_roc_overlay(models, split)
        overlay_pr  = plot_pr_overlay(models, split)
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
        # add metrics
        for name, (y, p) in models.items():
            fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
            ap = average_precision_score(y, p)
            ece, brier = _ece_brier(y, p)
            rows.append({
                "split": split, "model": f"{name} [metrics]",
                "roc_path": "", "pr_path": "", "reliability_path": "",
                "auc": round(roc_auc,3), "pr_auc": round(ap,3),
                "ece": round(ece,3), "brier": round(brier,4),
             })
        if args.with_calibration and models:
            plot_reliability_calibrated_for_all(models, split)

        if args.with_groupwise and models:
            g = _maybe_groups(split)
            if g is not None:
                for name, (y, p) in models.items():
                    for gid in np.unique(g):
                        m = (g == gid)
                        if m.sum() < 50:
                            continue
                        ece, brier = _ece_brier(y[m], p[m])
                        rows.append({
                            "split": split, "model": f"{name} [group={int(gid)}]",
                            "roc_path": "", "pr_path": "", "reliability_path": "",
                            "auc": None, "pr_auc": None,
                            "ece": round(ece,3), "brier": round(brier,4),
                        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTDIR, "plot_index.csv"), index=False)
        print("✅ Plots saved to", OUTDIR)
    else:
        print("No plots generated.")

    if args.with_groupwise and not os.path.exists(os.path.join(PROBDIR, "groups_val.npy")):
        print("💡 Tip: add group ids at results/probs/groups_val.npy (aligned to *_val_*.npy) for group-wise reliability.")


if __name__ == "__main__":
    main()
