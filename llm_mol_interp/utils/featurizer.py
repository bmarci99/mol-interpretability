from rdkit import Chem
from rdkit.Chem import rdchem
import torch
from torch_geometric.data import Data
import numpy as np

ATOM_LIST = list(range(1, 119))



def atom_features(atom):
    Z = atom.GetAtomicNum()
    feats = [
        Z,
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(includeNeighbors=True),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetExplicitValence(),   # keep ONLY explicit (not implicit)
    ]
    hyb = [int(atom.GetHybridization()==h) for h in
           [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D]]
    feats += hyb
    return np.array(feats, dtype=np.int64)

def bond_features(bond: rdchem.Bond):
    bt = bond.GetBondType()
    feats = [
        int(bt==rdchem.BondType.SINGLE),
        int(bt==rdchem.BondType.DOUBLE),
        int(bt==rdchem.BondType.TRIPLE),
        int(bt==rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]
    return np.array(feats, dtype=np.float32)

def mol_to_pyg(mol: Chem.Mol, y: np.ndarray = None, mol_id: str = None):
    Chem.Kekulize(Chem.RemoveHs(mol), clearAromaticFlags=False)
    N = mol.GetNumAtoms()
    x_list = [atom_features(mol.GetAtomWithIdx(i)) for i in range(N)]
    x = torch.tensor(np.stack(x_list, 0), dtype=torch.long)

    src, dst, e_list = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        # undirected edges
        src += [i, j]; dst += [j, i]
        e_list += [bf, bf]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.stack(e_list, 0), dtype=torch.float32) if e_list else torch.zeros((0,6))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.z = x[:,0]  # atomic number for embedding
    data.node_feats = x[:,1:]  # the rest (categorical ints)
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32)
    data.mol_id = mol_id or ""
    return data
