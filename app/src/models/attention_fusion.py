# src/models/attention_fusion.py
from typing import List

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """
    Simple attention-based fusion of multiple modality embeddings.

    Inputs: list of tensors [B, D] (e.g., nutrition, text, graph, region).
    Output: fused tensor [B, D].
    """

    def __init__(self, dim: int, num_modalities: int):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities

        # one learnable query vector for "what matters" in this context
        self.query = nn.Parameter(torch.randn(dim))

        # projection to ensure all modalities live in the same space
        self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_modalities)])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        modalities: list of [B, D] tensors (same order as in __init__)
        """
        assert len(modalities) == self.num_modalities

        proj_mods = [layer(m) for layer, m in zip(self.proj, modalities)]  # each [B, D]
        stack = torch.stack(proj_mods, dim=1)  # [B, M, D]

        # attention scores: dot(query, modality)
        q = self.query.view(1, 1, -1)  # [1,1,D]
        scores = (stack * q).sum(dim=-1)  # [B, M]
        weights = self.softmax(scores)  # [B, M]

        # weighted sum over modalities
        weights = weights.unsqueeze(-1)  # [B, M, 1]
        fused = (stack * weights).sum(dim=1)  # [B, D]
        return fused
