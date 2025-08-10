# run_train_transformer.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from src.model_transformer import SponsorRiskTransformer
from src.prepare_sequences import build_sequences_rich_trends
from src.train.metrics import compute_auc_pr, best_f1_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 1) Load sequences (numeric 9ch + trends)
    X, y, lengths, sponsors = build_sequences_rich_trends("data/aact_extracted.csv", max_seq_len=10)
    print("Input dims:", X.shape)
    # label sanity
    y_np = y.numpy()
    uniq, cnts = np.unique(y_np, return_counts=True)
    print("Label distribution:", dict(zip(uniq.tolist(), cnts.tolist())))
    if len(uniq) < 2:
        raise SystemExit("❗ Single-class labels. Check mapping or data.")

    # 2) Frozen split
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[va_idx], y[va_idx]

    # 3) Model
    model = SponsorRiskTransformer(
        input_dim=X.shape[2], d_model=64, nhead=4, num_layers=2,
        dim_feedforward=128, dropout=0.1
    ).to(DEVICE)

    # 4) DataLoaders (consistent with GRU runner)
    bs = 128
    tr_ds = TensorDataset(X_train, y_train)
    va_ds = TensorDataset(X_val,   y_val)
    tr_ld = DataLoader(tr_ds, batch_size=bs, sampler=RandomSampler(tr_ds), pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=bs, sampler=SequentialSampler(va_ds), pin_memory=True)

    # 5) Loss/optim/sched (pos_weight on the SAME device)
    pos_w = torch.tensor((y_train == 0).sum().item() / max((y_train == 1).sum().item(), 1), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    # 6) Train (simple early stopping on val logloss)
    best_state, best_logloss, patience, PATIENCE = None, float("inf"), 0, 5
    EPOCHS = 15

    from src.train import infer_lengths_from_padding
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            L = infer_lengths_from_padding(xb)
            logits = model(xb, L)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validate
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                L = infer_lengths_from_padding(xb)
                p = torch.sigmoid(model(xb, L))
                all_p.append(p.cpu()); all_y.append(yb.cpu())
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
    out_path = "sponsorsrisk_transformer.pt"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved model to {out_path}")
    auc, pr = compute_auc_pr(y_va, p_va)
    thr, acc_thr, f1_thr = best_f1_threshold(y_va, p_va)
    print(f"Val AUC: {auc:.3f} | PR-AUC: {pr:.3f} | Best-F1: {f1_thr:.3f} @ thr={thr:.3f}")
