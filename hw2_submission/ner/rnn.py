# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: BDG83

import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from ner.nn.module import Module


class RNN(Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        super().__init__()

        assert num_layers > 0

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        logging.info(f"no shared weights across layers")

        nonlinearity_dict = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        if nonlinearity not in nonlinearity_dict:
            raise ValueError(f"{nonlinearity} not supported, choose one of: [tanh, relu, prelu]")
        self.nonlinear = nonlinearity_dict[nonlinearity]

        self.W = nn.Linear(embedding_dim, hidden_dim)
        
        self.Uks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.Wks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])

        # self.U = nn.Linear(hidden_dim, hidden_dim)

        self.V = nn.Linear(hidden_dim, output_dim)

        self.apply(self.init_weights)

    def _initial_hidden_states(
        self, batch_size: int, init_zeros: bool = False, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        if init_zeros:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        else:
            hidden_states = nn.init.xavier_normal_(
                torch.empty(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        return list(map(torch.squeeze, hidden_states))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        
        z_prev = self._initial_hidden_states(len(embeddings), True, embeddings.device)

        (batch_size, batch_max_length, embedding_dim) = embeddings.shape
        
        all_y = []

        for t in range(batch_max_length):
          xt = embeddings[:,t,:]

          z_prime = self.W(xt)
          z_prime2 = self.Uks[0](z_prev[0])

          z = self.nonlinear(z_prime + z_prime2)
            
          z_prev[0] = z

          for l in range(self.num_layers-1):
            hidden_prime = self.Wks[l](z)
            hidden_prime2 = self.Uks[l](z_prev[l+1])

            z = self.nonlinear(hidden_prime + hidden_prime2)
            z_prev[l+1] = z

          y = self.V(z)

          all_y.append(y)

        return torch.stack(all_y, dim=1)



        






        
