# scripts/serve_inference.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.inference.pipeline import InferencePipeline
from src.inference.featurize import featurize_history

app = FastAPI()
pipe = InferencePipeline(bundle_dir="models", model_type="combined")

class Trial(BaseModel):
    start_date: str
    phase: str
    enrollment: int
    allocation: str
    masking: str
    primary_purpose: str
    intervention_types: str
    overall_status: str

class SponsorHistory(BaseModel):
    sponsor_name: str
    trials: List[Trial]

@app.post("/predict")
def predict(history: SponsorHistory):
    Xn, Xc, L = featurize_history([t.model_dump() for t in history.trials], max_seq_len=10)
    probs, labels = pipe.predict((Xn, Xc, L))
    return {"sponsor_name": history.sponsor_name, "probability": probs[0], "label": int(labels[0])}
