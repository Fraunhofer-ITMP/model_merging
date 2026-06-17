from __future__ import annotations

import torch
from torch import nn


class SolubilityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
