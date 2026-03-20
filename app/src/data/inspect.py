import pandas as pd
import os
from src.config import PROCESSED_DIR

files = [
    "diet1_raw.parquet",
    "diet2_raw.parquet",
    "diet3_raw.parquet",
    "nutrition_raw.parquet",
    "safety_raw.parquet",
    "users_raw.parquet"
]

print("\n=== Inspect processed parquet files ===")
for f in files:
    path = os.path.join(PROCESSED_DIR, f)
    if os.path.exists(path):
        print(f"\n>>> {f}")
        df = pd.read_parquet(path)
        print(df.shape)
        print(df.columns)
    else:
        print(f"[WARN] Missing: {f}")

print("\n=== DONE ===")
