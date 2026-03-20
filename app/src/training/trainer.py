# src/training/trainer.py
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import PROCESSED_DIR
from src.training.config import (
    EMBEDDING_DIM,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    NEGATIVE_SAMPLES,
    RANDOM_SEED,
)

USER_INTERACTIONS_PATH = os.path.join(PROCESSED_DIR, "user_interactions.parquet")

# model artifacts directory (under src/models_artifacts)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- DATASET ----------

class InteractionDataset(Dataset):
    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        num_items: int,
        negative_samples: int,
    ):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.num_items = num_items
        self.negative_samples = negative_samples

        # adjacency set: which items a user has already interacted with
        self.user_pos_items: Dict[int, set] = {}
        for u, i in zip(user_indices, item_indices):
            self.user_pos_items.setdefault(int(u), set()).add(int(i))

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int):
        u = int(self.user_indices[idx])
        i_pos = int(self.item_indices[idx])

        users = [u]
        items = [i_pos]
        labels = [1.0]

        # sample negatives
        for _ in range(self.negative_samples):
            while True:
                j = np.random.randint(0, self.num_items)  # 0..num_items-1
                if j not in self.user_pos_items.get(u, set()):
                    break
            users.append(u)
            items.append(j)
            labels.append(0.0)

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )


# ---------- MODEL ----------

class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, dim)
        self.item_factors = nn.Embedding(num_items, dim)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_factors(user_idx)   # [..., D]
        v = self.item_factors(item_idx)   # [..., D]
        scores = (u * v).sum(dim=-1)      # [...]
        return scores


# ---------- HELPERS ----------

def _load_interactions() -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    df = pd.read_parquet(USER_INTERACTIONS_PATH)

    # Expecting columns: user_id, food_id
    required = {"user_id", "food_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"user_interactions.parquet missing required columns: {missing}"
        )

    # Drop invalid (NaN) user_id or food_id
    df = df.dropna(subset=["user_id", "food_id"])

    # Cast to int
    df["user_id"] = df["user_id"].astype(int)
    df["food_id"] = df["food_id"].astype(int)

    # Get unique valid IDs
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["food_id"].unique())

    # Build mappings
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(item_ids)}

    # Keep only rows mappable to both dicts
    df = df[df["user_id"].isin(user2idx)]
    df = df[df["food_id"].isin(item2idx)]

    # Now map to indices
    df["user_idx"] = df["user_id"].map(user2idx).astype(int)
    df["item_idx"] = df["food_id"].map(item2idx).astype(int)

    # Final safety check
    assert df["user_idx"].between(0, len(user2idx)-1).all(), "Bad user_idx mapping!"
    assert df["item_idx"].between(0, len(item2idx)-1).all(), "Bad item_idx mapping!"

    return df, user2idx, item2idx


# ---------- TRAINING ----------

def train_matrix_factorization():
    print("=== TRAINER: loading interactions and embeddings ===")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df, user2idx, item2idx = _load_interactions()
    num_users = len(user2idx)
    num_items = len(item2idx)

    # Build dataset from unique positive (user, item) pairs
    positives = df[["user_idx", "item_idx"]].drop_duplicates()
    user_idx_arr = positives["user_idx"].values.astype(int)
    item_idx_arr = positives["item_idx"].values.astype(int)

    print(f"  [dataset] num_positive_pairs={len(user_idx_arr)}")
    print(
        f"  [dataset] item_idx stats: min={item_idx_arr.min()}, "
        f"max={item_idx_arr.max()}"
    )

    dataset = InteractionDataset(
        user_idx_arr,
        item_idx_arr,
        num_items=num_items,
        negative_samples=NEGATIVE_SAMPLES,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        dim=EMBEDDING_DIM,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print("=== TRAINING matrix factorization model ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for users, items, labels in dataloader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(users, items)  # shape [B*(1+neg),]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"  epoch {epoch}/{NUM_EPOCHS} - loss={avg_loss:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), MODEL_DIR / "mf_model.pt")

    # helper to invert mapping
    def _invert(d: Dict[int, int]) -> Dict[int, int]:
        return {v: k for k, v in d.items()}

    np.save(MODEL_DIR / "user2idx.npy", user2idx)
    np.save(MODEL_DIR / "item2idx.npy", item2idx)
    np.save(MODEL_DIR / "idx2user.npy", _invert(user2idx))
    np.save(MODEL_DIR / "idx2item.npy", _invert(item2idx))

    print(f"Saved model + mappings to {MODEL_DIR}")


def main():
    train_matrix_factorization()


if __name__ == "__main__":
    main()
