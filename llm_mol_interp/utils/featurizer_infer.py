# llm_mol_interp/utils/featurizer_infer.py
from typing import Optional, Dict, Any
import numpy as np, torch
from rdkit import Chem
from torch_geometric.data import Data

# import the SAME primitive feature builders you used at training:
from llm_mol_interp.utils.featurizer import atom_features, bond_features  # your existing functions

def mol_to_pyg_from_config(mol: Chem.Mol,
                           y: np.ndarray | None = None,
                           mol_id: str | None = None,
                           cfg: Optional[Dict[str, Any]] = None) -> Data:
    """
    Reproduce train-time featurization using flags in cfg.
    If cfg is None, defaults to what TRAINING used (fill below).
    """
    cfg = cfg or {}

    # Example toggles (adapt to your atom_features implementation):
    # Use cfg keys to disable/enable parts so the final len==expected
    # For example:
    # use_chirality = bool(cfg.get("use_chirality", False))
    # use_formal_charge = bool(cfg.get("use_formal_charge", False))
    # atom_feature_set = cfg.get("atom_feature_set", "minimal")  # e.g., "minimal"→44-d

    # IMPORTANT: your atom_features() already encodes all parts; if it
    # doesn't accept flags, create variants here to drop columns to match len.
    Chem.Kekulize(Chem.RemoveHs(mol), clearAromaticFlags=False)
    N = mol.GetNumAtoms()

    # Build atom features with the SAME pipeline as training
    x_list = [atom_features(mol.GetAtomWithIdx(i)) for i in range(N)]
    x = torch.tensor(np.stack(x_list, 0), dtype=torch.long)

    # Build bonds
    src, dst, e_list = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]; dst += [j, i]
        e_list += [bf, bf]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.stack(e_list, 0), dtype=torch.float32) if e_list else torch.zeros((0,6))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.z = x[:,0]
    data.node_feats = x[:,1:]
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32)
    data.mol_id = mol_id or ""
    return data
