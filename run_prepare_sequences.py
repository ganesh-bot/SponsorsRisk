# run_prepare_sequences.py

from src.prepare_sequences import build_sequences_rich

X, y = build_sequences_rich("data/aact_extracted.csv")

print("✅ Sequence preparation complete")
print("Input shape (X):", X.shape)
print("Target shape (y):", y.shape)
