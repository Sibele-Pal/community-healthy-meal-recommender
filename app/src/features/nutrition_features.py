# src/features/nutrition_features.py
import os
import pandas as pd
import numpy as np

from src.config import PROCESSED_DIR

MASTER_FOOD_CATALOG = os.path.join(PROCESSED_DIR, "master_food_catalog.parquet")
NUTRITION_FEATURES_PATH = os.path.join(PROCESSED_DIR, "item_nutrition_features.parquet")

# main nutrient columns (if some are missing, we auto-skip)
NUTRI_COLS = [
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
]

def build_nutrition_features() -> None:
    print("=== NUTRITION FEATURES: building numeric nutrient matrix ===")
    df = pd.read_parquet(MASTER_FOOD_CATALOG)
    if "food_id" not in df.columns:
        raise KeyError("master_food_catalog.parquet must contain a 'food_id' column.")

    cols = [c for c in NUTRI_COLS if c in df.columns]
    if not cols:
        raise ValueError(
            f"None of the expected nutrient columns {NUTRI_COLS} "
            "were found in master_food_catalog.parquet."
        )

    X = df[cols].fillna(0.0).astype(float)

    # standardize (z-score)
    mean = X.mean(axis=0)
    std = X.std(axis=0).replace(0, 1.0)
    X_norm = (X - mean) / std

    feat_df = pd.DataFrame(
        X_norm.values,
        columns=[f"nutri_{i}" for i in range(X_norm.shape[1])]
    )
    feat_df.insert(0, "food_id", df["food_id"].values)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feat_df.to_parquet(NUTRITION_FEATURES_PATH, index=False)
    print(
        f"Saved nutrition features to {NUTRITION_FEATURES_PATH} "
        f"with shape {feat_df.shape}"
    )


if __name__ == "__main__":
    build_nutrition_features()
