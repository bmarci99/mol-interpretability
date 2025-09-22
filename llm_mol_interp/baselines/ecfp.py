# llm_mol_interp/baselines/ecfp.py
from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def smiles_to_ecfp_bits(smiles_list, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute ECFP bit vectors for a list of SMILES.
    Returns an array of shape [N, n_bits] with dtype uint8 (0/1).
    Invalid SMILES rows get all-zeros.
    """
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(str(smi))
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i] = (arr > 0).astype(np.uint8)
        except Exception:
            # keep zeros
            pass
    return X