# diagnostics_deep_attribution.py
import os, numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.prepare_sequences import build_sequences_rich
from src.model import SponsorRiskGRU
from src.model_transformer import SponsorRiskTransformer
from src.train import infer_lengths_from_padding

OUTDIR = "results/diagnostics_deep"
os.makedirs(OUTDIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_split():
    tr = np.load("splits/train_idx.npy")
    va = np.load("splits/val_idx.npy")
    return tr, va

def load_data():
    X, y, lengths, sponsors = build_sequences_rich("data/aact_extracted.csv", max_seq_len=10)
    return X, y, lengths, sponsors

def train_or_load_gru(Xtr, ytr, Xva, yva):
    model = SponsorRiskGRU(input_dim=Xtr.shape[2], hidden_dim=64, num_layers=1, dropout=0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    import torch.nn.functional as F
    pos = (ytr==1).sum().item(); neg = (ytr==0).sum().item()
    pos_w = torch.tensor(neg/max(pos,1), dtype=torch.float32, device=DEVICE)
    best_auc, best = -1.0, None

    tr_ds = torch.utils.data.TensorDataset(Xtr, ytr)
    va_ds = torch.utils.data.TensorDataset(Xva, yva)
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=256)

    for ep in range(1, 8):  # 7 quick epochs
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            L = infer_lengths_from_padding(xb)
            logits = model(xb, L)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()

        # quick val AUC
        model.eval(); allp, ally = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                L = infer_lengths_from_padding(xb)
                p = torch.sigmoid(model(xb, L))
                allp.append(p.cpu()); ally.append(yb.cpu())
        y_true = torch.cat(ally).numpy()
        y_prob = torch.cat(allp).numpy()
        try: auc = roc_auc_score(y_true, y_prob)
        except: auc = float("nan")
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc; best = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)
    return model

def train_or_load_tx(Xtr, ytr, Xva, yva):
    model = SponsorRiskTransformer(input_dim=Xtr.shape[2], d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    import torch.nn.functional as F
    pos = (ytr==1).sum().item(); neg = (ytr==0).sum().item()
    pos_w = torch.tensor(neg/max(pos,1), dtype=torch.float32, device=DEVICE)
    best_auc, best = -1.0, None

    tr_ds = torch.utils.data.TensorDataset(Xtr, ytr)
    va_ds = torch.utils.data.TensorDataset(Xva, yva)
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=256)

    for ep in range(1, 8):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            L = infer_lengths_from_padding(xb)
            logits = model(xb, L)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); allp, ally = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                L = infer_lengths_from_padding(xb)
                p = torch.sigmoid(model(xb, L))
                allp.append(p.cpu()); ally.append(yb.cpu())
        y_true = torch.cat(ally).numpy()
        y_prob = torch.cat(allp).numpy()
        try: auc = roc_auc_score(y_true, y_prob)
        except: auc = float("nan")
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc; best = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)
    return model

def grad_saliency(model, Xb, yb):
    """
    Mean |∂logit/∂x| over val set, per (timestep, feature).
    CuDNN RNNs require training mode for backward, so we temporarily switch to train().
    We also disable dropout to keep saliency stable.
    """
    model_mode_before = model.training
    model.train()
    # try to disable dropout modules if present
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p, m.inplace = 0.0, False

    grads_sum = torch.zeros_like(Xb[0:1])  # (1, T, D)
    n = 0
    for i in range(0, len(Xb), 128):
        xb = Xb[i:i+128].to(DEVICE).clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        L = infer_lengths_from_padding(xb)
        logit = model(xb, L)  # (B,)
        # use sum of logits so every sample contributes gradient
        logit.sum().backward()
        grads = xb.grad.detach().abs().sum(dim=0, keepdim=True)  # (1, T, D)
        grads_sum += grads.cpu()
        n += xb.size(0)

    # restore original mode
    if not model_mode_before:
        model.eval()
    sal = grads_sum / max(n, 1)
    return sal.squeeze(0).numpy()  # (T, D)


def feature_ablation(model, Xva, yva, which="feature", idx=0):
    """AUC drop by masking one feature channel or one timestep."""
    base_auc = auc_of(model, Xva, yva)
    Xmasked = Xva.clone()
    if which == "feature":
        Xmasked[:,:,idx] = 0.0
        tag = f"feat{idx}"
    else:
        Xmasked[:,idx,:] = 0.0
        tag = f"time{idx}"
    drop = base_auc - auc_of(model, Xmasked, yva)
    return tag, base_auc, drop

def auc_of(model, Xva, yva):
    model.eval(); allp = []
    with torch.no_grad():
        for i in range(0, len(Xva), 256):
            xb = Xva[i:i+256].to(DEVICE)
            L = infer_lengths_from_padding(xb)
            p = torch.sigmoid(model(xb, L))
            allp.append(p.cpu())
    y_prob = torch.cat(allp).numpy()
    try:
        return roc_auc_score(yva.numpy(), y_prob)
    except:
        return float("nan")

if __name__ == "__main__":
    print("Loading data/splits...")
    X, y, lengths, sponsors = load_data()
    tr_idx, va_idx = load_split()
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    print("Training small GRU & Transformer (few epochs for diagnostics)...")
    gru = train_or_load_gru(Xtr, ytr, Xva, yva)
    tx  = train_or_load_tx (Xtr, ytr, Xva, yva)

    # ---- Saliency maps ----
    print("Computing gradient saliency maps...")
    sal_gru = grad_saliency(gru, Xva, yva)  # (T,D)
    sal_tx  = grad_saliency(tx,  Xva, yva)

    # Plot per-feature saliency over timesteps
    feat_names = ["phase_enc","enroll_z","gap_months"]
    for name, sal in [("GRU", sal_gru), ("Transformer", sal_tx)]:
        plt.figure()
        for d in range(sal.shape[1]):
            plt.plot(sal[:,d], label=feat_names[d])
        plt.legend()
        plt.title(f"{name} saliency by timestep (validation)")
        plt.xlabel("timestep (oldest → newest)")
        plt.ylabel("|∂logit/∂x| (avg)")
        plt.savefig(os.path.join(OUTDIR, f"{name.lower()}_saliency_by_time.png"))
        plt.close()

    # ---- Simple ablations ----
    print("Running ablations (mask one feature / one timestep)...")
    rows = []
    for mdl_name, mdl in [("GRU",gru),("Transformer",tx)]:
        # by feature
        for d in range(X.shape[2]):
            tag, base, drop = feature_ablation(mdl, Xva.clone(), yva, which="feature", idx=d)
            rows.append([mdl_name, "feature", tag, base, drop])
        # by timestep (test last 5 timesteps)
        T = X.shape[1]
        test_steps = list(range(max(0,T-5), T))
        for t in test_steps:
            tag, base, drop = feature_ablation(mdl, Xva.clone(), yva, which="time", idx=t)
            rows.append([mdl_name, "timestep", tag, base, drop])

    import pandas as pd
    df_ablate = pd.DataFrame(rows, columns=["model","type","mask","base_auc","auc_drop"])
    df_ablate.to_csv(os.path.join(OUTDIR,"ablations.csv"), index=False)
    print("✅ Saved:",
          f"{OUTDIR}/gru_saliency_by_time.png, {OUTDIR}/transformer_saliency_by_time.png, {OUTDIR}/ablations.csv")
