# src/features/text_features.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.config import MASTER_FOOD_CATALOG_PARQUET, TEXT_FEATURES_PATH

TEXT_COL = "food_name"   # we treat this as the main text field


def build_text_features():
    print("=== TEXT FEATURES: TF-IDF + SVD embeddings ===")
    print(f"Using text column: {TEXT_COL}")

    # ---------- load master catalog ----------
    df = pd.read_parquet(MASTER_FOOD_CATALOG_PARQUET)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in master_food_catalog")

    texts = df[TEXT_COL].fillna("").astype(str)

    # ---------- TF-IDF ----------
    # keep settings simple and robust
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,           # was probably too strict before
        max_features=5000
    )
    tfidf = vectorizer.fit_transform(texts)
    n_samples, n_features = tfidf.shape
    print(f"TF-IDF shape = {tfidf.shape}")

    # ---------- SVD (robust to low feature count) ----------
    if n_features < 2:
        # SVD requires at least 2 features; fall back to raw TF-IDF as dense
        print(
            "[WARN] TF-IDF has only 1 feature; skipping SVD and "
            "using the raw TF-IDF column as 1-D embedding."
        )
        reduced = tfidf.toarray()
    else:
        n_components = min(64, n_features - 1)
        print(f"Using TruncatedSVD with n_components = {n_components}")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(tfidf)

    # ---------- build DataFrame ----------
    emb_dim = reduced.shape[1]
    emb_cols = [f"text_emb_{i}" for i in range(emb_dim)]
    text_df = pd.DataFrame(reduced, columns=emb_cols)
    text_df["food_id"] = df["food_id"].values

    # ---------- save ----------
    TEXT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    text_df.to_parquet(TEXT_FEATURES_PATH, index=False)
    print(f"Saved text features to {TEXT_FEATURES_PATH} with shape {text_df.shape}")
    print("=== TEXT FEATURES COMPLETE ===")


if __name__ == "__main__":
    build_text_features()
