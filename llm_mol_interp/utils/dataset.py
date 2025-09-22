import pandas as pd, numpy as np
from rdkit import Chem
from torch_geometric.data import InMemoryDataset
from .featurizer import mol_to_pyg

class Tox21Dataset(InMemoryDataset):
    def __init__(self, csv_path, smiles_col, task_cols):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.smiles = df[smiles_col].tolist()
        Y = df[task_cols].to_numpy()
        self.num_tasks = len(task_cols)
        self.data_list = []
        for i, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(smi)
            y = Y[i].astype(np.float32)
            y[np.isnan(y)] = -1.0
            y = y[None, :]  # <-- make it [1, num_tasks] so batches become [B, num_tasks]
            self.data_list.append(mol_to_pyg(mol, y, mol_id=str(i)))
    def get(self, idx): return self.data_list[idx]
    def __len__(self): return len(self.data_list)
