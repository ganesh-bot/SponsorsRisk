# run_train_combined.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from src.models.combined import CombinedGRU
from src.features.prepare_sequences import build_sequences_with_cats_trends
from src.train.metrics import compute_auc_pr, best_f1_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 1) Load numeric + categorical sequences (+ lengths + vocab sizes)
    seq_out = build_sequences_with_cats_trends("data/aact_extracted.csv", max_seq_len=10, verbose=False)
    # tolerate extra returns (e.g., vocab maps)
    Xn, Xc, y, L, sponsors, vocab_sizes, *extra = seq_out
    print("Numeric dims:", Xn.shape, "| Cat dims:", Xc.shape)

    # 2) Frozen split
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")
    Xn_tr, Xn_va = Xn[tr_idx], Xn[va_idx]
    Xc_tr, Xc_va = Xc[tr_idx], Xc[va_idx]
    y_tr,  y_va  = y[tr_idx],  y[va_idx]
    L_tr,  L_va  = L[tr_idx],  L[va_idx]

    # 3) Model
    model = CombinedGRU(
        num_dim=Xn.shape[2],
        cat_vocab_sizes=vocab_sizes,
        emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1
    ).to(DEVICE)

    # 4) DataLoaders (tuple dataset: (Xn, Xc, L, y))
    bs = 128
    tr_ds = TensorDataset(Xn_tr, Xc_tr, L_tr, y_tr)
    va_ds = TensorDataset(Xn_va, Xc_va, L_va, y_va)
    tr_ld = DataLoader(tr_ds, batch_size=bs, sampler=RandomSampler(tr_ds), pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=bs, sampler=SequentialSampler(va_ds), pin_memory=True)

    # 5) Loss/optim/sched (pos_weight on SAME device)
    pos_w = torch.tensor((y_tr == 0).sum().item() / max((y_tr == 1).sum().item(), 1), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    # 6) Train (early stop on val logloss)
    best_state, best_logloss, patience, PATIENCE = None, float("inf"), 0, 5
    EPOCHS = 15
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xn, xc, l, yb in tr_ld:
            xn, xc, l, yb = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yb.to(DEVICE)
            logits = model(xn, xc, l)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xn, xc, l, yv in va_ld:
                xn, xc, l, yv = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yv.to(DEVICE)
                p = torch.sigmoid(model(xn, xc, l))
                all_p.append(p.cpu()); all_y.append(yv.cpu())
        p_va = torch.cat(all_p).numpy()
        y_va = torch.cat(all_y).numpy()

        eps = 1e-7
        val_logloss = -np.mean(y_va * np.log(p_va + eps) + (1 - y_va) * np.log(1 - p_va + eps))
        sched.step(val_logloss)

        if val_logloss < best_logloss:
            best_logloss, best_state, patience = val_logloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Save + report unified metrics
    out_path = "sponsorsrisk_combined.pt"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved model to {out_path}")
    auc, pr = compute_auc_pr(y_va, p_va)
    thr, acc_thr, f1_thr = best_f1_threshold(y_va, p_va)
    print(f"Val AUC: {auc:.3f} | PR-AUC: {pr:.3f} | Best-F1: {f1_thr:.3f} @ thr={thr:.3f}")
