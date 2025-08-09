import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from src.prepare_sequences import build_sequences_with_cats_trends
from src.model_combined import CombinedGRU
from src.train import infer_lengths_from_padding

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

def train_combined_get_probs(Xn_tr, Xc_tr, y_tr, L_tr,
                             Xn_va, Xc_va, y_va, L_va, vocab_sizes,
                             epochs=30, lr=1e-3, batch=256, pos_weight=None):
    model = CombinedGRU(num_dim=Xn_tr.shape[2], cat_vocab_sizes=vocab_sizes,
                        emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
    tr_ds = TensorDataset(Xn_tr, Xc_tr, L_tr, y_tr)
    va_ds = TensorDataset(Xn_va, Xc_va, L_va, y_va)
    tr_ld = DataLoader(tr_ds, batch_size=batch, sampler=RandomSampler(tr_ds), pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch, sampler=SequentialSampler(va_ds), pin_memory=True)

    best_auc, best_state, pat, PATIENCE = -1.0, None, 0, 5
    for ep in range(1, epochs+1):
        model.train()
        for xn, xc, l, yb in tr_ld:
            xn, xc, l, yb = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yb.to(DEVICE)
            logits = model(xn, xc, l)
            loss = F.binary_cross_entropy_with_logits(
                logits, yb,
                pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # quick val for scheduler + early stopping
        model.eval(); allp, ally = [], []
        with torch.no_grad():
            for xn, xc, l, yb in va_ld:
                xn, xc, l, yb = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yb.to(DEVICE)
                p = torch.sigmoid(model(xn, xc, l))
                allp.append(p.cpu()); ally.append(yb.cpu())
        y_true = torch.cat(ally).numpy(); p_va = torch.cat(allp).numpy()

        # logloss for scheduler
        eps = 1e-7
        val_logloss = -np.mean(y_true*np.log(p_va+eps) + (1-y_true)*np.log(1-p_va+eps))
        sched.step(val_logloss)

        # AUC for early stopping (threshold-free)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, p_va)
        except Exception:
            auc = -1.0

        if auc > best_auc:
            best_auc, best_state, pat = auc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final train/val probs for calibration
    def infer_probs(Xn, Xc, L, batch=512):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch):
                xn, xc, l = Xn[i:i+batch].to(DEVICE), Xc[i:i+batch].to(DEVICE), L[i:i+batch].to(DEVICE)
                out.append(torch.sigmoid(model(xn, xc, l)).cpu())
        return torch.cat(out).numpy(), model

    p_tr, model = infer_probs(Xn_tr, Xc_tr, L_tr)
    p_va, _     = infer_probs(Xn_va, Xc_va, L_va)
    return model, p_tr, p_va

def best_f1_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.r_[0.0, t]
    f1 = 2 * (p*r) / np.maximum(p + r, 1e-9)
    k = int(np.nanargmax(f1))
    return float(t[k])

if __name__ == "__main__":
    # 1) load data & split
    tr_idx = np.load("splits/train_idx.npy"); va_idx = np.load("splits/val_idx.npy")
    Xn, Xc, y, L, sponsors, vocab_sizes, = None, None, None, None, None, None

    Xn, Xc, y, L, sponsors, vocab_sizes, vocab_maps = build_sequences_with_cats_trends(
        "data/aact_extracted.csv", max_seq_len=10, verbose=False
    )
    Xn_tr, Xn_va = Xn[tr_idx], Xn[va_idx]
    Xc_tr, Xc_va = Xc[tr_idx], Xc[va_idx]
    y_tr, y_va   = y[tr_idx],  y[va_idx]
    L_tr, L_va   = L[tr_idx],  L[va_idx]

    pos_w = torch.tensor((y_tr==0).sum().item()/max((y_tr==1).sum().item(),1), dtype=torch.float32) * 1.2

    # 2) train combined + get probs
    model, p_tr, p_va = train_combined_get_probs(
        Xn_tr, Xc_tr, y_tr, L_tr, Xn_va, Xc_va, y_va, L_va, vocab_sizes,
        epochs=30, lr=1e-3, batch=256, pos_weight=pos_w
    )

    # 3) fit isotonic on train probs
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr.numpy())

    # 4) choose best-F1 threshold on calibrated val
    p_va_iso = iso.transform(p_va)
    thr = best_f1_threshold(y_va.numpy(), p_va_iso)

    # 5) save artifacts
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "sponsorsrisk_combined.pt"))
    with open(os.path.join(MODELS_DIR, "calibrator_combined_isotonic.pkl"), "wb") as f:
        pickle.dump(iso, f)

    with open(os.path.join(MODELS_DIR, "thresholds.json"), "w") as f:
        json.dump({"Combined-9+4": thr}, f, indent=2)

    meta = {
        "model": "Combined-9+4",
        "num_dim": int(Xn.shape[2]),
        "cat_dims": {k: len(v) for k, v in vocab_maps.items()},
        "max_seq_len": 10,
        "seed": SEED,
        "device": DEVICE,
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Persist vocabularies for inference mapping
    with open(os.path.join(MODELS_DIR, "vocab.json"), "w") as f:
        json.dump(vocab_maps, f, indent=2)

    print(f"✅ Exported model bundle to '{MODELS_DIR}'\n- sponsorsrisk_combined.pt\n- calibrator_combined_isotonic.pkl\n- thresholds.json\n- meta.json\n- vocab.json")
