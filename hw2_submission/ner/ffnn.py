# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: BDG83

from dataclasses import dataclass
from datasets.utils.py_utils import zip_dict

import torch
import torch.nn.functional as F
from torch import nn

from ner.nn.module import Module


class FFNN(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1) -> None:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.ffnn.html."""
        super().__init__()

        assert num_layers > 0

        # Initialize W & V

        # W : X (L x d+1) --> Z (L x h)
        self.W = nn.Linear(embedding_dim, hidden_dim)

        # Hidden to Hidden 
        self.Uks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])

        # V : Z (L x h+1) --> Y (L x o)
        self.V = nn.Linear(hidden_dim, output_dim)

        self.apply(self.init_weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.ffnn.html."""
        
        z_prime = self.W(embeddings)
        z = F.relu(z_prime)

        for Uk in self.Uks:
          z = Uk(z)
          z = F.relu(z)

        y_prime = self.V(z)

        return y_prime
