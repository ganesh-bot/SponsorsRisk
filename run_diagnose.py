# run_diagnose.py
import pandas as pd

df = pd.read_csv("data/aact_extracted.csv")

print("unique overall_status values:")
print(df["overall_status"].dropna().str.strip().str.lower().value_counts())

print("\nmissing sponsor_name rows:", df["sponsor_name"].isna().sum())
print("total rows:", len(df))
