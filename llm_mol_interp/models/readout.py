# llm_mol_interp/models/readout.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch_scatter import scatter_add

class GatedAttentionPool(nn.Module):
    """
    AttentiveFP-style gated attention readout:
      a_i = sigmoid(W_g h_i) * tanh(W_h h_i)
      pool = sum_i a_i
    """
    def __init__(self, hidden, gate_hidden=None):
        super().__init__()
        gate_hidden = gate_hidden or hidden
        self.Wg = nn.Linear(hidden, gate_hidden)
        self.Wh = nn.Linear(hidden, gate_hidden)

    def forward(self, h_atom, batch):
        g = torch.sigmoid(self.Wg(h_atom))
        h = torch.tanh(self.Wh(h_atom))
        a = g * h                            # [N, H]
        pooled = scatter_add(a, batch, dim=0)
        return pooled
