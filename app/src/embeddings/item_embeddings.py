# src/embeddings/item_embeddings.py
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

MASTER_FOOD_CATALOG = os.path.join(PROCESSED_DIR, "master_food_catalog.parquet")
NUTRITION_FEATURES_PATH = os.path.join(PROCESSED_DIR, "item_nutrition_features.parquet")
TEXT_FEATURES_PATH = os.path.join(PROCESSED_DIR, "item_text_features.parquet")
CENTRALITY_PATH = os.path.join(PROCESSED_DIR, "item_centrality.parquet")
ITEM_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "item_embeddings.parquet")


def build_item_embeddings() -> None:
    print("=== ITEM EMBEDDINGS: combining nutrition + text + centrality ===")

    master = pd.read_parquet(MASTER_FOOD_CATALOG)[["food_id"]].drop_duplicates()
    nutri = pd.read_parquet(NUTRITION_FEATURES_PATH)
    text = pd.read_parquet(TEXT_FEATURES_PATH)

    # centrality is optional but recommended
    if os.path.exists(CENTRALITY_PATH):
        central = pd.read_parquet(CENTRALITY_PATH)
        # expect columns: food_id, degree, betweenness, pagerank
        central_cols = [c for c in central.columns if c != "food_id"]
    else:
        central = pd.DataFrame({"food_id": master["food_id"]})
        central_cols = []

    df = master.merge(nutri, on="food_id", how="left")
    df = df.merge(text, on="food_id", how="left")
    df = df.merge(central, on="food_id", how="left")

    df = df.fillna(0.0)

    feature_cols = [c for c in df.columns if c != "food_id"]
    X = df[feature_cols].astype(float).values

    # L2 normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    emb_df = pd.DataFrame(X_norm, columns=[f"emb_{i}" for i in range(X_norm.shape[1])])
    emb_df.insert(0, "food_id", df["food_id"].values)

    emb_df.to_parquet(ITEM_EMBEDDINGS_PATH, index=False)
    print(
        f"Saved item embeddings to {ITEM_EMBEDDINGS_PATH} "
        f"with shape {emb_df.shape}"
    )


def load_item_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    """Helper used by models / serving."""
    df = pd.read_parquet(ITEM_EMBEDDINGS_PATH)
    item_ids = df["food_id"].values
    X = df[[c for c in df.columns if c.startswith("emb_")]].values.astype(float)
    return df, X


if __name__ == "__main__":
    build_item_embeddings()
