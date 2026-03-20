# src/models/transformer_seq.py
from typing import Optional

import torch
import torch.nn as nn


class SequenceTransformerModel(nn.Module):
    """
    Lightweight Transformer encoder for user food sequences.

    - item_ids:   [B, T] LongTensor of item indices (0..num_items-1 or PAD index)
    - region_ids: [B, T] LongTensor of region indices (same T as items)

    We separate:
      - encode(...)  -> returns hidden states [B, T, D]
      - forward(...) -> returns logits [B, T, num_items]
    so that other modules (like SeqRecModel) can reuse the encoder only.
    """

    def __init__(
        self,
        num_items: int,
        num_regions: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 100,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_regions = num_regions
        self.d_model = d_model
        self.max_len = max_len

        # embeddings
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.region_embedding = nn.Embedding(num_regions, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # prediction head: predict next item id
        self.output_layer = nn.Linear(d_model, num_items)

    def encode(
        self,
        item_ids: torch.LongTensor,
        region_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Return encoded hidden states [B, T, D] without prediction head."""
        B, T = item_ids.shape
        device = item_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(pos)

        item_emb = self.item_embedding(item_ids)

        if region_ids is not None:
            region_emb = self.region_embedding(region_ids)
        else:
            region_emb = torch.zeros_like(item_emb)

        x = item_emb + region_emb + pos_emb  # [B, T, D]
        x = self.encoder(x)                  # [B, T, D]
        return x

    def forward(
        self,
        item_ids: torch.LongTensor,
        region_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Standard forward: logits over items [B, T, num_items]."""
        x = self.encode(item_ids, region_ids)         # [B, T, D]
        logits = self.output_layer(x)                # [B, T, num_items]
        return logits
