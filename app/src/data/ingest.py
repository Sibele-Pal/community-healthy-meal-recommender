# src/data/ingest.py
import pandas as pd
from pathlib import Path

from src.config import (
    FINAL_DIET_CSV,
    MEAL_SUGGESTIONS_CSV,
    MICRO_MACRO_CSV,
    NUTRITION_CSV,
    FOOD_SAFETY_CSV,
    USERS_CSV,
    MERGED_MASTER_CSV,
    DIET_FINAL_PARQUET,
    DIET_MEAL_PARQUET,
    DIET_MICRO_PARQUET,
    NUTRITION_PARQUET,
    FOOD_SAFETY_PARQUET,
    USERS_PARQUET,
    MERGED_MASTER_PARQUET,
    PROCESSED_DIR,
)


def safe_read_csv(path: Path, label: str):
    """Read a CSV if it exists; otherwise warn and return None."""
    if not path.exists():
        print(f"[WARN] {label}: file not found at {path}")
        return None
    print(f"Loading {label} ({path})")
    df = pd.read_csv(path)
    print(f"  shape = {df.shape}")
    return df


def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("=== INGEST: loading raw CSVs and saving to parquet ===")

    diet_final = safe_read_csv(FINAL_DIET_CSV, "final_diet.csv")
    diet_meal = safe_read_csv(MEAL_SUGGESTIONS_CSV, "meal_suggestions.csv")
    diet_micro = safe_read_csv(MICRO_MACRO_CSV, "micro_macro_nutrients.csv")

    nutrition = safe_read_csv(NUTRITION_CSV, "nutrition_cleaned_ready.csv")
    food_safety = safe_read_csv(FOOD_SAFETY_CSV, "rasff_cleaned_ready.csv")
    users = safe_read_csv(USERS_CSV, "synthetic_users_cleaned_ready.csv")
    merged_master = safe_read_csv(
        MERGED_MASTER_CSV, "master_food_safety_recommender.csv"
    )

    if diet_final is not None:
        diet_final.to_parquet(DIET_FINAL_PARQUET, index=False)
    if diet_meal is not None:
        diet_meal.to_parquet(DIET_MEAL_PARQUET, index=False)
    if diet_micro is not None:
        diet_micro.to_parquet(DIET_MICRO_PARQUET, index=False)

    if nutrition is not None:
        nutrition.to_parquet(NUTRITION_PARQUET, index=False)
    if food_safety is not None:
        food_safety.to_parquet(FOOD_SAFETY_PARQUET, index=False)
    if users is not None:
        users.to_parquet(USERS_PARQUET, index=False)
    if merged_master is not None:
        merged_master.to_parquet(MERGED_MASTER_PARQUET, index=False)

    print("=== INGEST COMPLETE ===")


if __name__ == "__main__":
    run()
