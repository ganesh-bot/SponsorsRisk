# run_train_combined.py
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.prepare_sequences import build_sequences_with_cats
from src.model_combined import CombinedGRU
from src.train import fit

if __name__ == "__main__":
    X_num, X_cat, y, lengths, sponsors, vocab_sizes = build_sequences_with_cats(
        "data/aact_extracted.csv",
        max_seq_len=10
    )

    y_np = y.numpy()
    uniq, cnts = np.unique(y_np, return_counts=True)
    print("Label distribution:", dict(zip(uniq.tolist(), cnts.tolist())))
    if len(uniq) < 2:
        raise SystemExit("❗ Single-class labels. Check mapping or data.")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = np.array(sponsors)
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(y_np)), y_np, groups))

    Xn_tr, Xc_tr, y_tr, L_tr = X_num[tr_idx], X_cat[tr_idx], y[tr_idx], lengths[tr_idx]
    Xn_va, Xc_va, y_va, L_va = X_num[va_idx], X_cat[va_idx], y[va_idx], lengths[va_idx]

    model = CombinedGRU(
        num_dim=X_num.shape[2],        # 3
        cat_vocab_sizes=vocab_sizes,   # 4 categorical fields
        emb_dim=16,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1
    )

    # Wrap a small adapter to pass (X_num, X_cat) to fit()
    # We'll modify fit-callsite to lambda the forward.
    from src.train import make_loaders, evaluate, class_weight_from_labels

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    pos_weight = class_weight_from_labels(y_tr).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # build loaders of tuples (X_num, X_cat, y)
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    train_ds = TensorDataset(Xn_tr, Xc_tr, L_tr, y_tr)
    val_ds   = TensorDataset(Xn_va, Xc_va, L_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=128, sampler=RandomSampler(train_ds))
    val_loader   = DataLoader(val_ds,   batch_size=128, sampler=SequentialSampler(val_ds))

    import torch.nn.functional as F
    best_auc, best_state, patience = -1.0, None, 3
    for ep in range(1, 16):
        # train
        model.train()
        tr_loss = 0.0
        for Xn, Xc, L, yt in train_loader:
            Xn, Xc, L, yt = Xn.to(device), Xc.to(device), L.to(device), yt.to(device)
            logits = model(Xn, Xc, L)
            loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
            optim.zero_grad(); loss.backward(); optim.step()
            tr_loss += loss.item() * yt.size(0)

        # eval
        model.eval()
        all_y, all_p, va_loss = [], [], 0.0
        with torch.no_grad():
            for Xn, Xc, L, yv in val_loader:
                Xn, Xc, L, yv = Xn.to(device), Xc.to(device), L.to(device), yv.to(device)
                logits = model(Xn, Xc, L)
                loss = F.binary_cross_entropy_with_logits(logits, yv)
                probs = torch.sigmoid(logits)
                va_loss += loss.item() * yv.size(0)
                all_y.append(yv.cpu()); all_p.append(probs.cpu())

        y_true = torch.cat(all_y).numpy()
        y_prob = torch.cat(all_p).numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        print(f"[Epoch {ep:02d}] train_loss={tr_loss/len(train_ds):.4f} | "
              f"val_loss={va_loss/len(val_ds):.4f} | acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 3
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), "sponsorsrisk_combined.pt")
    print(f"✅ Saved model to sponsorsrisk_combined.pt")
    print(f"Best validation AUC: {best_auc:.3f}")
