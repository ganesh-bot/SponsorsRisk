# run_prepare_sequences.py

from prepare_sequences import build_sequences

X, y = build_sequences("data/aact_extracted.csv")

print("✅ Sequence preparation complete")
print("Input shape (X):", X.shape)
print("Target shape (y):", y.shape)
