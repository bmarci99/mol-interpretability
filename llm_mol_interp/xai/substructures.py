from rdkit import Chem
from collections import defaultdict

from rdkit import Chem
# you already have rdkit_alerts_smarts() in this file

def detect_substructures(mol: Chem.Mol, include=None) -> list[dict]:
    """
    Simple programmatic detector: match a library of SMARTS on the molecule.
    Returns a list of dicts: {"name": str, "atom_indices": [int], "smarts": str}
    - include: optional list of catalog names if your rdkit_alerts_smarts supports it
    """
    lib = rdkit_alerts_smarts(include)   # -> {name: smarts}
    out = []
    for name, smarts in lib.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            matches = mol.GetSubstructMatches(patt)
        except Exception:
            matches = ()
        if matches:
            atoms = sorted({i for m in matches for i in m})
            out.append({"name": name, "atom_indices": atoms, "smarts": smarts})
    return out


# Hand-picked core alerts
BASE_SMARTS = {
    "aromatic nitro (charged)": "[c;a][N+](=O)[O-]",
    "aromatic nitro":           "[c;a][N](=O)=O",
    "primary aromatic amine":   "[NX3H2][c;a]",
    "nitrosamine":              "[NX3]-N(=O)",
    "epoxide":                  "C1OC1",
    "michael acceptor (enone)": "C=CC=O",
    "quinone":                  "O=C1C=CC(=O)C=C1",
    "aldehyde":                 "[CX3H1](=O)[#6]",
    "azo":                      "N=N",
    "haloarene":                "[c;a][F,Cl,Br,I]",
    "polycyclic aromatic":      "c1ccc2cccc2c1",
}

# Try to import RDKit catalogs; if not available, we’ll just use BASE_SMARTS
try:
    from .alerts_lib import rdkit_alerts_smarts
except Exception:
    rdkit_alerts_smarts = None

_SMARTS_CACHE = None

def _get_smarts_library():
    global _SMARTS_CACHE
    if _SMARTS_CACHE is not None:
        return _SMARTS_CACHE
    lib = {}
    # Load RDKit catalogs if possible
    if rdkit_alerts_smarts is not None:
        try:
            lib.update(rdkit_alerts_smarts())
        except Exception as e:
            print("[WARN] RDKit FilterCatalog not available, using only base SMARTS:", e)
    # Always include base set
    lib.update(BASE_SMARTS)
    _SMARTS_CACHE = lib
    return _SMARTS_CACHE

def _matches(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    return mol.GetSubstructMatches(patt)

def select_alert_by_overlap(mol, important_bonds, thr=0.4, mode="hybrid"):
    """
    Return dict {name, atom_indices, score, faithful} or None.
    mode: 'attr' | 'presence' | 'hybrid' (prefer attr, fallback to presence)
    """
    from collections import defaultdict
    SMARTS = _get_smarts_library()

    # Build atom importance from bond importances
    atom_w = defaultdict(float)
    for i, j, w in important_bonds:
        atom_w[i] += float(w); atom_w[j] += float(w)

    # Normalize to [0,1], pick top-K atoms for stability
    top_atoms = set()
    if atom_w:
        maxw = max(atom_w.values()) or 1.0
        for k in list(atom_w.keys()):
            atom_w[k] /= maxw
        N = mol.GetNumAtoms()
        K = max(5, int(0.3 * N))
        top_atoms = set(sorted(atom_w, key=atom_w.get, reverse=True)[:K])

    best_name, best_match, best_score = None, None, -1.0
    for name, sm in SMARTS.items():
        for match in _matches(mol, sm):
            mset = set(match)
            score = 0.0
            if atom_w:
                score = sum(atom_w.get(a, 0.0) for a in match) / float(len(match))
                if not (mset & top_atoms):
                    score *= 0.5  # penalize if it misses top atoms
            if score > best_score:
                best_name, best_match, best_score = name, match, score

    if best_name is not None and ((best_score >= thr) or mode != "attr"):
        return {"name": best_name, "atom_indices": list(best_match),
                "score": float(best_score), "faithful": best_score >= thr}

    if mode == "hybrid":
        # presence-only fallback
        for name, sm in SMARTS.items():
            m = _matches(mol, sm)
            if m:
                return {"name": name, "atom_indices": list(m[0]), "score": 0.0, "faithful": False}
    return None
