from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from src.inference.pipeline import load_bundle, predict_from_sponsor_df

app = FastAPI(title="SponsorsRisk Inference API", version="1.0.0")
model, calibrator, thresholds, vocab_maps, meta = load_bundle(models_dir="models")

class TrialRecord(BaseModel):
    start_date: str
    phase: Optional[str] = None
    enrollment: Optional[float] = None
    allocation: Optional[str] = None
    masking: Optional[str] = None
    primary_purpose: Optional[str] = None
    intervention_types: Optional[str] = None
    overall_status: Optional[str] = None  # helpful for trends; if unknown, pass "unknown"

class SponsorHistory(BaseModel):
    sponsor_name: Optional[str] = None
    trials: List[TrialRecord]

@app.get("/health")
def health():
    return {"status": "ok", "model": "Combined-9+4", "device": str(meta.get("device", "cpu"))}

@app.post("/predict")
def predict(payload: SponsorHistory):
    # Convert trials list to DataFrame
    df = pd.DataFrame([t.dict() for t in payload.trials])
    result = predict_from_sponsor_df(df, model, calibrator, thresholds, vocab_maps)
    return {"sponsor_name": payload.sponsor_name, **result}
