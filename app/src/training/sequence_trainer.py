# src/training/sequence_trainer.py
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import (
    PROCESSED_DIR,
    MASTER_FOOD_CATALOG_PARQUET,
)
from src.models.transformer_seq import SequenceTransformerModel
from src.models.attention_fusion import AttentionFusion
from src.training.config import (
    EMBEDDING_DIM,   # we reuse as d_model
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
)

# paths
USER_INTERACTIONS_PATH = os.path.join(PROCESSED_DIR, "user_interactions.parquet")

# where to save this model
MODEL_DIR = Path(__file__).resolve().parents[1] / "models_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- DATASET ----------

class SequenceDataset(Dataset):
    """
    Each item is one (user, sequence, regions, target item).
    sequence: list[int] of item_idx (no PAD inside)
    regions:  list[int] of region_idx aligned with sequence
    target:   int item_idx
    user_idx: int
    """

    def __init__(
        self,
        sequences: List[List[int]],
        region_seqs: List[List[int]],
        targets: List[int],
        user_idxs: List[int],
    ):
        assert len(sequences) == len(targets) == len(user_idxs) == len(region_seqs)
        self.sequences = sequences
        self.region_seqs = region_seqs
        self.targets = targets
        self.user_idxs = user_idxs

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.region_seqs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.user_idxs[idx], dtype=torch.long),
        )


def make_collate_fn(num_items_with_pad: int):
    """
    Collate variable-length sequences into padded tensors.
    We reserve index (num_items_with_pad - 1) for PAD.
    Region pad index is 0.
    """
    pad_item_idx = num_items_with_pad - 1

    def collate(batch):
        seqs, reg_seqs, targets, users = zip(*batch)
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        B = len(seqs)

        # [B, T]
        item_tensor = torch.full(
            (B, max_len), pad_item_idx, dtype=torch.long
        )
        region_tensor = torch.zeros((B, max_len), dtype=torch.long)

        for i, (s, r) in enumerate(zip(seqs, reg_seqs)):
            L = len(s)
            item_tensor[i, :L] = torch.tensor(s, dtype=torch.long)
            region_tensor[i, :L] = torch.tensor(r, dtype=torch.long)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        targets_tensor = torch.stack(targets)  # [B]
        users_tensor = torch.stack(users)      # [B]

        return item_tensor, region_tensor, targets_tensor, users_tensor, lengths_tensor

    return collate


# ---------- MODEL ----------

class SeqRecModel(nn.Module):
    """
    Sequence recommender:

    - encodes item sequence with SequenceTransformerModel
    - takes last hidden state (per user sequence)
    - fuses it with a static user preference embedding via AttentionFusion
    - outputs logits over items
    """

    def __init__(
        self,
        num_items_with_pad: int,
        num_regions: int,
        num_users: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 100,
    ):
        super().__init__()
        self.num_items_with_pad = num_items_with_pad
        self.num_regions = num_regions
        self.num_users = num_users
        self.d_model = d_model

        # underlying Transformer encoder
        self.encoder_model = SequenceTransformerModel(
            num_items=num_items_with_pad,
            num_regions=num_regions,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
        )

        # static user preference embedding
        self.user_pref = nn.Embedding(num_users, d_model)

        # attention-based fusion of [sequence_rep, user_pref]
        self.fusion = AttentionFusion(dim=d_model, num_modalities=2)

        # prediction head
        self.output_layer = nn.Linear(d_model, num_items_with_pad)

    def forward(
        self,
        item_seqs: torch.LongTensor,    # [B, T]
        region_seqs: torch.LongTensor,  # [B, T]
        user_idx: torch.LongTensor,     # [B]
        lengths: torch.LongTensor,      # [B]
    ) -> torch.Tensor:
        # encode full sequence: [B, T, D]
        enc = self.encoder_model.encode(item_seqs, region_seqs)
        B, T, D = enc.shape
        device = enc.device

        # last non-pad index per sequence
        lengths = lengths.to(device)
        last_idx = (lengths - 1).clamp(min=0)  # [B]
        idx = last_idx.view(B, 1, 1).expand(-1, 1, D)  # [B, 1, D]
        seq_rep = enc.gather(1, idx).squeeze(1)        # [B, D]

        # user preference embedding
        user_emb = self.user_pref(user_idx.to(device))  # [B, D]

        # attention fusion
        fused = self.fusion([seq_rep, user_emb])        # [B, D]

        logits = self.output_layer(fused)               # [B, num_items_with_pad]
        return logits


# ---------- BUILD SEQUENCES FROM DATA ----------

def build_sequence_data() -> Tuple[
    SequenceDataset,
    int,  # num_items_with_pad
    int,  # num_regions
    int,  # num_users
    Dict[int, int],  # user2idx
    Dict[int, int],  # foodid2idx (item)
    Dict[int, int],  # idx2foodid
    Dict[str, int],  # region2idx
]:
    # master catalog: food_id + region
    master = pd.read_parquet(MASTER_FOOD_CATALOG_PARQUET)
    if "food_id" not in master.columns:
        raise KeyError("master_food_catalog.parquet must contain 'food_id' column.")
    if "region" not in master.columns:
        raise KeyError("master_food_catalog.parquet must contain 'region' column.")

    master["food_id"] = master["food_id"].astype(int)
    master["region"] = master["region"].fillna("unknown").astype(str)

    # unique items
    food_ids = sorted(master["food_id"].unique())
    foodid2idx: Dict[int, int] = {fid: idx for idx, fid in enumerate(food_ids)}
    idx2foodid: Dict[int, int] = {idx: fid for fid, idx in foodid2idx.items()}
    num_items = len(food_ids)
    num_items_with_pad = num_items + 1  # last index reserved for PAD

    # region mapping
    unique_regions = sorted(master["region"].unique())
    # reserve 0 for pad/unknown, regions start at 1
    region2idx: Dict[str, int] = {"<pad>": 0}
    for r in unique_regions:
        if r not in region2idx:
            region2idx[r] = len(region2idx)
    num_regions = len(region2idx)

    # map food_id -> region_idx
    foodid2region_idx: Dict[int, int] = {}
    for row in master[["food_id", "region"]].itertuples(index=False):
        fid = int(row.food_id)
        reg = str(row.region)
        foodid2region_idx[fid] = region2idx.get(reg, 0)

    # interactions: user_id, food_id
    inter = pd.read_parquet(USER_INTERACTIONS_PATH)
    required = {"user_id", "food_id"}
    missing = required - set(inter.columns)
    if missing:
        raise KeyError(f"user_interactions.parquet missing columns: {missing}")

    inter = inter.dropna(subset=["user_id", "food_id"])
    inter["user_id"] = inter["user_id"].astype(int)
    inter["food_id"] = inter["food_id"].astype(int)

    # keep only items we know
    inter = inter[inter["food_id"].isin(foodid2idx)]

    # map item_idx + user_idx
    inter["item_idx"] = inter["food_id"].map(foodid2idx).astype(int)

    user_ids = sorted(inter["user_id"].unique())
    user2idx: Dict[int, int] = {uid: idx for idx, uid in enumerate(user_ids)}
    inter["user_idx"] = inter["user_id"].map(user2idx).astype(int)
    num_users = len(user2idx)

    # build per-user sequences (simple: all but last as input, last as target)
    sequences: List[List[int]] = []
    region_seqs: List[List[int]] = []
    targets: List[int] = []
    user_idxs: List[int] = []

    # sort interactions per user (by original order)
    inter = inter.sort_values(["user_idx"]).reset_index(drop=True)

    for u_idx, grp in inter.groupby("user_idx"):
        items = grp["item_idx"].tolist()
        fids = grp["food_id"].tolist()
        if len(items) < 2:
            continue

        # input sequence (no pad, raw indices 0..num_items-1)
        input_items = items[:-1]
        target_item = items[-1]
        input_regions = [foodid2region_idx[int(fid)] for fid in fids[:-1]]

        sequences.append(input_items)
        region_seqs.append(input_regions)
        targets.append(target_item)
        user_idxs.append(int(u_idx))

    if not sequences:
        raise RuntimeError("No valid user sequences found (need at least 2 items per user).")

    dataset = SequenceDataset(sequences, region_seqs, targets, user_idxs)
    return (
        dataset,
        num_items_with_pad,
        num_regions,
        num_users,
        user2idx,
        foodid2idx,
        idx2foodid,
        region2idx,
    )


# ---------- TRAINING LOOP ----------

def train_sequence_model():
    print("=== SEQ TRAINER: building sequences ===")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    (
        dataset,
        num_items_with_pad,
        num_regions,
        num_users,
        user2idx,
        foodid2idx,
        idx2foodid,
        region2idx,
    ) = build_sequence_data()

    print(f"  num_users={num_users}, num_items_with_pad={num_items_with_pad}")
    print(f"  num_sequences={len(dataset)}, num_regions={num_regions}")

    # max sequence length for positional embeddings
    max_len = max(len(seq) for seq in dataset.sequences)

    collate_fn = make_collate_fn(num_items_with_pad)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqRecModel(
        num_items_with_pad=num_items_with_pad,
        num_regions=num_regions,
        num_users=num_users,
        d_model=EMBEDDING_DIM,
        n_heads=4,
        n_layers=2,
        max_len=max_len,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("=== TRAINING sequence Transformer + AttentionFusion model ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for item_seqs, region_seqs, targets, users, lengths in dataloader:
            item_seqs = item_seqs.to(device)
            region_seqs = region_seqs.to(device)
            targets = targets.to(device)
            users = users.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(item_seqs, region_seqs, users, lengths)  # [B, num_items_with_pad]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"  epoch {epoch}/{NUM_EPOCHS} - loss={avg_loss:.4f}")

    # ---- save artifacts ----
    torch.save(model.state_dict(), MODEL_DIR / "seq_model.pt")
    np.save(MODEL_DIR / "seq_user2idx.npy", user2idx)
    np.save(MODEL_DIR / "seq_foodid2idx.npy", foodid2idx)
    np.save(MODEL_DIR / "seq_idx2foodid.npy", idx2foodid)
    np.save(MODEL_DIR / "seq_region2idx.npy", region2idx)

    print(f"Saved sequence model + mappings to {MODEL_DIR}")
    print("=== SEQ TRAINER COMPLETE ===")


def main():
    train_sequence_model()


if __name__ == "__main__":
    main()
