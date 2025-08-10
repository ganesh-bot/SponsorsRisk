# scripts/plot_slices.py
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "results/plots"
os.makedirs(OUTDIR, exist_ok=True)

def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_sponsor_type(model_name="Combined-9+4"):
    df = pd.read_csv("results/slices_sponsor_type.csv")
    dfm = df[df["model"] == model_name].copy()
    if dfm.empty:
        print(f"[plot_slices] No rows for model='{model_name}' in slices_sponsor_type.csv")
        return ""
    dfm = dfm.sort_values("auc", ascending=False)
    fig = plt.figure()
    plt.barh(dfm["sponsor_type"], dfm["auc"])
    for i, (auc, n) in enumerate(zip(dfm["auc"], dfm["n"])):
        plt.text(auc + 0.002, i, f"n={n}", va="center", fontsize=8)
    plt.xlabel("AUC"); plt.ylabel("Sponsor type")
    plt.title(f"AUC by sponsor type ({model_name}, calibrated)")
    out = os.path.join(OUTDIR, f"slices_sponsor_type_{model_name.replace(' ','_')}.png")
    _save(fig, out)
    print("✅ Fig 4 saved:", out)
    return out

def plot_histlen(model_name="Combined-9+4"):
    df = pd.read_csv("results/slices_histlen.csv")
    dfm = df[df["model"] == model_name].copy()
    if dfm.empty:
        print(f"[plot_slices] No rows for model='{model_name}' in slices_histlen.csv")
        return ""
    order = ["1-2","3-5","6-10"]
    dfm["bucket"] = pd.Categorical(dfm["bucket"], categories=order, ordered=True)
    dfm = dfm.sort_values("bucket")
    fig = plt.figure()
    plt.bar(dfm["bucket"], dfm["auc"])
    for x, (auc, n) in zip(dfm["bucket"], zip(dfm["auc"], dfm["n"])):
        plt.text(x, auc + 0.002, f"n={n}", ha="center", fontsize=8)
    plt.xlabel("History length (trials)"); plt.ylabel("AUC")
    plt.title(f"AUC by history length ({model_name}, calibrated)")
    out = os.path.join(OUTDIR, f"slices_histlen_{model_name.replace(' ','_')}.png")
    _save(fig, out)
    print("✅ Fig 5 saved:", out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Combined-9+4")
    args = ap.parse_args()
    plot_sponsor_type(args.model)
    plot_histlen(args.model)

if __name__ == "__main__":
    main()
