# SponsorsRisk – Paper Pack

- Commit: `70fdd15`
- This folder contains the exact tables and figures used in the paper.

## Figures
- Figure 1 (ROC overlay): `roc_val_overlay.png`
- Figure 2 (PR overlay): `pr_val_overlay.png`
- Figure 3 (Reliability, uncal vs iso, Combined-9+4): `reliability_val_combined_uncal_vs_iso.png`
- Figure 4 (AUC by sponsor type, Combined-9+4): `slices_sponsor_type_Combined-9+4.png`
- Figure 5 (AUC by history length, Combined-9+4): `slices_histlen_Combined-9+4.png`

## Tables
- `metrics_overall.csv` / `metrics_overall_calibrated.csv` (AUC, PR-AUC, best-F1 & threshold)
- JSON mirrors of the above for programmatic use.

Re-generate with:
```bash
python scripts/build_paper_pack.py --split val --dpi 300 --out docs/paper_pack
```