import torch, torch.nn as nn, torch.nn.functional as F
from torch_scatter import scatter_add


class DMPNN(nn.Module):
    """
    Chemprop-like Directed MPNN
    - Directed edges (i->j)
    - Updates edge states h_{i->j} using incoming to i excluding reverse (j->i)
    - Returns logits, atom embeddings, edge embeddings
    """
    def __init__(self, atom_in_dim, bond_in_dim, hidden=256, depth=3, dropout=0.1, num_tasks=12):
        super().__init__()
        self.hidden, self.depth = hidden, depth
        self.embed_z = nn.Embedding(119, 32)

        # atom_in: [N, 32+atom_in_dim] → hidden
        self.atom_in = nn.Linear(atom_in_dim + 32, hidden)
        self.bond_in = nn.Linear(bond_in_dim, hidden)

        self.W_msg   = nn.Linear(hidden, hidden, bias=False)
        self.gru     = nn.GRUCell(hidden, hidden)

        self.dropout_p = dropout
        self.atom_out = nn.Linear(32 + atom_in_dim + hidden, hidden)
        self.readout  = nn.Linear(hidden, num_tasks)

        self.norm_e = nn.LayerNorm(hidden)
        self.norm_a = nn.LayerNorm(hidden)

    @staticmethod
    def _build_directed(edge_index, num_nodes):
        src_u, dst_u = edge_index
        src_dir = torch.cat([src_u, dst_u], dim=0)
        dst_dir = torch.cat([dst_u, src_u], dim=0)
        key = (src_dir.to(torch.long) << 32) + dst_dir.to(torch.long)
        rev_key = (dst_dir.to(torch.long) << 32) + src_dir.to(torch.long)
        order = torch.argsort(key)
        key_sorted = key[order]
        idx_in_sorted = torch.searchsorted(key_sorted, rev_key)
        rev = order[idx_in_sorted]
        map_dir_to_und = torch.cat([
            torch.arange(src_u.size(0), device=src_u.device),
            torch.arange(dst_u.size(0), device=dst_u.device)
        ], dim=0)

        return src_dir, dst_dir, rev, map_dir_to_und

    def forward(self, x_z, x_atom, edge_index, edge_attr, batch):
        N = x_atom.size(0)
        src_dir, dst_dir, rev, map_dir_to_und = self._build_directed(edge_index, N)

        # initial node representation
        a_embed = self.embed_z(x_z)                          # [N,32]
        a_all   = torch.cat([a_embed, x_atom.float()], dim=1)
        ###h_src   = self.atom_in(a_all)[src_dir]               # [E_dir,H]

        # Node projection
        a_embed = self.embed_z(x_z)  # [N, 32]
        a_all = torch.cat([a_embed, x_atom.float()], 1)  # [N, 32+F]
        h_all = self.atom_in(a_all)

        src_dir = src_dir.to(h_all.device, dtype=torch.long).contiguous()
        h_src = torch.index_select(h_all, 0, src_dir)  # instead of h_all[src_dir]

        # Edge projection
        e_all = self.bond_in(edge_attr.float())
        map_dir_to_und = map_dir_to_und.to(e_all.device, dtype=torch.long).contiguous()
        e_in = torch.index_select(e_all, 0, map_dir_to_und)  # instead of e_all[map_dir_to_und]

        # initial edge states
        #e_in = self.bond_in(edge_attr.float())[map_dir_to_und]
        h_e = F.relu(h_src + e_in)
        h_e = self.norm_e(h_e)


        # message passing
        for _ in range(self.depth):
            m_in_node = scatter_add(h_e, dst_dir, dim=0, dim_size=N)
            m_excl = m_in_node[src_dir] - h_e[rev]
            m = self.W_msg(m_excl)
            h_new = F.relu(h_e + m)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            h_e = self.gru(h_new, h_e)
            h_e = self.norm_e(h_e)

        # aggregate to atoms
        m_to_atom = scatter_add(h_e, dst_dir, dim=0, dim_size=N)
        h_atom = F.relu(self.atom_out(torch.cat([a_all, m_to_atom], dim=1)))
        h_atom = self.norm_a(h_atom)

        # graph-level
        h_graph = scatter_add(h_atom, batch, dim=0)
        out = self.readout(F.dropout(h_graph, p=self.dropout_p, training=self.training))
        return out, h_atom, h_e
