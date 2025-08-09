# debug_cohort.py
import pandas as pd
from src.utils_data import VALID_PHASES

FAIL = {"terminated", "withdrawn", "suspended"}
SUCCESS = {"completed"}

df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])

print("Total trials:", len(df))
print("Unique sponsors:", df["sponsor_name"].nunique())

# 1) Sponsors with >= 2 trials
s2 = df.groupby("sponsor_name").size()
print("Sponsors with >= 2 trials:", (s2 >= 2).sum())

# 2) Interventional only
df_i = df[df["study_type"] == "Interventional"]
print("After Interventional filter:", df_i["sponsor_name"].nunique())

# 3) Keep only valid phases
df_ip = df_i[df_i["phase"].isin(VALID_PHASES)]
print("After Interventional + phase filter:", df_ip["sponsor_name"].nunique())

# 4) Sponsors with >= 2 trials AFTER this filter
s2_ip = df_ip.groupby("sponsor_name").size()
print("Multi-trial sponsors after filter:", (s2_ip >= 2).sum())
