# src/inference/pipeline.py
import os, json, joblib, torch
from src.model import SponsorRiskGRU
from src.model_transformer import SponsorRiskTransformer
from src.model_combined import CombinedGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAP = {
    "gru": SponsorRiskGRU,
    "transformer": SponsorRiskTransformer,
    "combined": CombinedGRU
}

class InferencePipeline:
    def __init__(self, bundle_dir, model_type):
        self.bundle_dir = bundle_dir
        self.model_type = model_type
        self.meta = json.load(open(os.path.join(bundle_dir, "meta.json")))
        self.thresholds = json.load(open(os.path.join(bundle_dir, "thresholds.json")))
        self.calibrator = joblib.load(os.path.join(bundle_dir, "calibrator_isotonic.pkl"))

        model_cls = MODEL_MAP[model_type]
        if model_type == "combined":
            self.model = model_cls(num_dim=self.meta["num_dim"], cat_vocab_sizes=self.meta["cat_vocab_sizes"],
                                   emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1)
        elif model_type == "gru":
            self.model = model_cls(input_dim=self.meta["input_dim"], hidden_dim=64, num_layers=1, dropout=0.1)
        elif model_type == "transformer":
            self.model = model_cls(input_dim=self.meta["input_dim"], d_model=64, nhead=4, num_layers=2,
                                   dim_feedforward=128, dropout=0.1)

        self.model.load_state_dict(torch.load(os.path.join(bundle_dir, f"sponsorsrisk_{model_type}.pt"), map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, data_tuple, threshold: float | None = None):
        """data_tuple matches training input (Combined: Xn, Xc, L) else (X, L)."""
        with torch.no_grad():
            if self.model_type == "combined":
                Xn, Xc, L = data_tuple
                prob = torch.sigmoid(self.model(Xn.to(DEVICE), Xc.to(DEVICE), L.to(DEVICE)))
            else:
                X, L = data_tuple
                prob = torch.sigmoid(self.model(X.to(DEVICE), L.to(DEVICE)))
        prob = prob.cpu().numpy().ravel()
        prob_cal = self.calibrator.transform(prob)
        th = self.thresholds.get("threshold") if threshold is None else float(threshold)
        label = (prob_cal >= th).astype(int)
        return prob_cal.tolist(), label.tolist()
