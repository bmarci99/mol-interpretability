from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
import os, json, torch, inspect
from torch_geometric.data import Data
from rdkit import Chem

from llm_mol_interp.models.dmpnn import DMPNN
from llm_mol_interp.utils.featurizer import mol_to_pyg  # uses your atom_features/bond_features

# ---------- helpers ----------

def _load_blob(ckpt: str) -> Any:
    return torch.load(ckpt, map_location="cpu")

def _extract_state(blob: Any) -> Dict[str, torch.Tensor]:
    if isinstance(blob, dict) and isinstance(blob.get("model"), dict):
        return blob["model"]
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob if isinstance(blob, dict) else {}

def _read_side(ckpt: str, blob: Any) -> Dict[str, Any]:
    side = {}
    base = os.path.dirname(os.path.abspath(ckpt))
    for name in ("hparams.json","args.json","config.json"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            try:
                side.update(json.load(open(p,"r")))
            except Exception:
                pass
    if isinstance(blob, dict) and isinstance(blob.get("hparams"), dict):
        side.update(blob["hparams"])
    return side

def _ensure_batch(data: Data) -> Data:
    import torch as _t
    if not hasattr(data, "batch") or data.batch is None:
        n = int(data.z.size(0)) if hasattr(data, "z") else int(data.x.size(0))
        data.batch = _t.zeros(n, dtype=_t.long)
    return data

def _force_dims(data: Data, atom_dim: int, bond_dim: int) -> Data:
    import torch as _t
    if hasattr(data, "node_feats"):
        cur = data.node_feats.size(1)
        if cur != atom_dim:
            if cur > atom_dim:
                data.node_feats = data.node_feats[:, :atom_dim].contiguous()
            else:
                pad = _t.zeros(data.node_feats.size(0), atom_dim - cur, dtype=data.node_feats.dtype)
                data.node_feats = _t.cat([data.node_feats, pad], dim=1)
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
        cur_e = data.edge_attr.size(1)
        if cur_e != bond_dim:
            if cur_e > bond_dim:
                data.edge_attr = data.edge_attr[:, :bond_dim].contiguous()
            else:
                pad = _t.zeros(data.edge_attr.size(0), bond_dim - cur_e, dtype=data.edge_attr.dtype)
                data.edge_attr = _t.cat([data.edge_attr, pad], dim=1)
    return data

def _build_model_from_state(state: Dict[str, torch.Tensor], side: Dict[str, Any]) -> DMPNN:
    # dims from checkpoint tensors
    atom_in_dim = int(state["atom_in.weight"].shape[1]) - 32  # RIGHT (44 - 32 = 12)
    bond_in_dim = int(state["bond_in.weight"].shape[1])  # 6
    num_tasks = int(state["readout.weight"].shape[0])  # 12
    hidden = int(side.get("hidden_dim", side.get("hidden", 256)))
    depth = int(side.get("num_gnn_layers", side.get("depth", 3)))
    dropout = float(side.get("dropout", 0.1))
    # construct EXACT ctor your class defines
    model = DMPNN(atom_in_dim=atom_in_dim,
                  bond_in_dim=bond_in_dim,
                  hidden=hidden,
                  depth=depth,
                  dropout=dropout,
                  num_tasks=num_tasks)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[DMPEngine] load_state_dict: missing={missing} unexpected={unexpected}")
    model.eval()
    return model

# ---------- engine ----------

class DMPEngine:
    def __init__(self, ckpt_paths: List[str], target_names: Optional[List[str]] = None):
        if not ckpt_paths:
            raise ValueError("No checkpoints provided")

        # build first to lock dims/tasks/config
        blob0 = _load_blob(ckpt_paths[0])
        state0 = _extract_state(blob0)
        if "atom_in.weight" not in state0 or "bond_in.weight" not in state0 or "readout.weight" not in state0:
            raise RuntimeError("Checkpoint missing expected keys: atom_in/bond_in/readout")

        side0 = _read_side(ckpt_paths[0], blob0)
        self.model_template = _build_model_from_state(state0, side0)

        # cache expected dims for featurization
        self.expected_atom_dim = int(state0["atom_in.weight"].shape[1])  # 44
        self.expected_bond_dim = int(state0["bond_in.weight"].shape[1])  # 6
        self.num_tasks = int(state0["readout.weight"].shape[0])          # 12

        # names
        if target_names and len(target_names) == self.num_tasks:
            self.target_names = target_names
        else:
            if target_names and len(target_names) != self.num_tasks:
                print(f"[DMPEngine] Warning: got {len(target_names)} names, but checkpoint has {self.num_tasks} outputs.")
            self.target_names = [f"task_{i}" for i in range(self.num_tasks)]

        # build ensemble models
        self.models = []
        for p in ckpt_paths:
            blob = _load_blob(p)
            state = _extract_state(blob)
            side = _read_side(p, blob)
            m = _build_model_from_state(state, side)
            self.models.append(m)

    def _featurize(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Bad SMILES: {smiles}")
        d = mol_to_pyg(mol)  # your featurizer

        # Enforce node_feats to 11 (not 44!), because x_atom = [z | node_feats] must be 12
        import torch as _t
        if hasattr(d, "node_feats"):
            cur = d.node_feats.size(1)
            if cur != 11:
                if cur > 11:
                    d.node_feats = d.node_feats[:, :11].contiguous()
                else:
                    pad = _t.zeros(d.node_feats.size(0), 11 - cur, dtype=d.node_feats.dtype)
                    d.node_feats = _t.cat([d.node_feats, pad], dim=1)

        # Enforce bond features to 6 (optional guard)
        if hasattr(d, "edge_attr") and d.edge_attr is not None and d.edge_attr.numel() > 0:
            cur_e = d.edge_attr.size(1)
            if cur_e != 6:
                if cur_e > 6:
                    d.edge_attr = d.edge_attr[:, :6].contiguous()
                else:
                    pad = _t.zeros(d.edge_attr.size(0), 6 - cur_e, dtype=d.edge_attr.dtype)
                    d.edge_attr = _t.cat([d.edge_attr, pad], dim=1)

        # ensure batch
        n = int(d.z.size(0)) if hasattr(d, "z") else int(d.x.size(0))
        d.batch = _t.zeros(n, dtype=_t.long)
        return d

    def _forward_one(self, model: DMPNN, d: Data) -> torch.Tensor:
        import torch as _t
        x_atom_full = _t.cat([d.z.view(-1, 1).long(), d.node_feats.long()], dim=1)  # [N, 12]
        logits, _, _ = model(d.z.long(), x_atom_full, d.edge_index, d.edge_attr, d.batch)
        return torch.sigmoid(logits).view(-1)

    def predict_proba(self, smiles: str) -> Dict[str, float]:
        d = self._featurize(smiles)
        with torch.no_grad():
            outs = [self._forward_one(m, d) for m in self.models]
        mean = torch.stack(outs).mean(dim=0)
        return {n: float(v) for n, v in zip(self.target_names, mean.tolist())}

    def predict_proba_with_mask(self, smiles: str, mask_atoms: List[int], mode: str = "zero") -> Dict[str, float]:
        import torch as _t
        d = self._featurize(smiles)
        if mask_atoms:
            n = int(d.z.size(0))
            mask = _t.zeros(n, dtype=_t.bool); mask[mask_atoms] = True
            # zero z and node_feats and drop edges touching masked atoms
            d.z = d.z.clone(); d.z[mask] = 0
            d.node_feats = d.node_feats.clone(); d.node_feats[mask] = 0
            ei = d.edge_index; keep = ~(mask[ei[0]] | mask[ei[1]])
            d.edge_index = ei[:, keep]
            if d.edge_attr is not None and d.edge_attr.numel() > 0:
                d.edge_attr = d.edge_attr[keep]
        with torch.no_grad():
            outs = [self._forward_one(m, d) for m in self.models]
        mean = torch.stack(outs).mean(dim=0)
        return {n: float(v) for n, v in zip(self.target_names, mean.tolist())}
