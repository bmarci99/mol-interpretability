from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random

def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return ""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core, isomericSmiles=True)

def scaffold_split(smiles_list, frac=(0.8, 0.1, 0.1), seed=0,
                   min_val=None, min_test=None):
    """
    Balanced scaffold split (grouped by Bemis–Murcko scaffolds).
    Greedy bin-packing by relative fill toward targets. Guarantees decent val/test sizes.

    min_val/min_test: absolute minimum #molecules for val/test (defaults to frac*len)
    """
    rng = random.Random(seed)
    buckets = {}
    for idx, smi in enumerate(smiles_list):
        key = murcko_scaffold(smi)
        buckets.setdefault(key, []).append(idx)

    # sort scaffolds by descending size (pack big groups first)
    groups = sorted(buckets.values(), key=len, reverse=True)

    N = len(smiles_list)
    tgt_train = int(frac[0] * N)
    tgt_val   = int(frac[1] * N)
    tgt_test  = N - tgt_train - tgt_val  # keep exact total

    if min_val  is None: min_val  = max(1, int(0.08 * N))
    if min_test is None: min_test = max(1, int(0.08 * N))

    train, val, test = [], [], []
    n_train = n_val = n_test = 0

    def rel_fill(n, tgt):  # smaller is "more room"
        return (n / max(tgt, 1e-9))

    for grp in groups:
        # choose the split with smallest relative fill
        cands = [("train", n_train, tgt_train),
                 ("val",   n_val,   tgt_val),
                 ("test",  n_test,  tgt_test)]
        cands.sort(key=lambda x: rel_fill(x[1], x[2]))
        best = cands[0][0]
        if best == "train":
            train += grp; n_train += len(grp)
        elif best == "val":
            val   += grp; n_val   += len(grp)
        else:
            test  += grp; n_test  += len(grp)

    # Post-fix: ensure minimum sizes by stealing from the biggest split
    def steal(dst_list, dst_n, min_needed):
        nonlocal train, val, test, n_train, n_val, n_test
        if dst_n >= min_needed: return
        # move whole scaffold groups from the largest split until reaching min_needed
        # rebuild groups-with-split
        split_map = {}
        for grp in groups:
            if set(grp).issubset(set(train)): split_map.setdefault("train", []).append(grp)
            elif set(grp).issubset(set(val)): split_map.setdefault("val", []).append(grp)
            else: split_map.setdefault("test", []).append(grp)
        def total(s): return sum(len(g) for g in split_map.get(s, []))
        # order donors by current size desc, then by largest group first
        donors = sorted(["train","val","test"], key=lambda s: total(s), reverse=True)
        donors = [d for d in donors if d != dst_list and total(d) > 0]

        for d in donors:
            # move largest donor group first
            split_map[d].sort(key=len, reverse=True)
            while split_map[d] and dst_n < min_needed:
                grp = split_map[d].pop(0)
                if d == "train":
                    for i in grp: train.remove(i)
                    n_train -= len(grp)
                elif d == "val":
                    for i in grp: val.remove(i)
                    n_val   -= len(grp)
                else:
                    for i in grp: test.remove(i)
                    n_test  -= len(grp)

                if dst_list == "val":
                    val += grp; n_val += len(grp); dst_n = n_val
                else:
                    test += grp; n_test += len(grp); dst_n = n_test
                if dst_n >= min_needed: break

    steal("val",  n_val,  min_val)
    steal("test", n_test, min_test)

    return train, val, test
