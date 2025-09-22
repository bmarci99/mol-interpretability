#!/usr/bin/env python
import os, json, argparse
from tqdm import tqdm
from rdkit import Chem

# uses your working wrapper
from predictor import DMPEngine

def load_jsonl(p):
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def write_jsonl(items, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

def per_atom_mask_importance(eng: DMPEngine, smiles: str, endpoint: str, topk: int = 10, mode: str = "zero"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []
    nA = mol.GetNumAtoms()

    p0 = eng.predict_proba(smiles)[endpoint]

    drops = []
    for i in range(nA):
        pm = eng.predict_proba_with_mask(smiles, mask_atoms=[i], mode=mode)[endpoint]
        drops.append(max(0.0, p0 - pm))

    mx = max(drops) if drops else 0.0
    imps = [d / mx if mx > 0 else 0.0 for d in drops]

    order = sorted(range(nA), key=lambda i: drops[i], reverse=True)
    top_atoms = [i for i in order[:min(topk, nA)] if drops[i] > 0]

    return imps, top_atoms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-names", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--mask-mode", default="zero", choices=["zero","delete"])
    args = ap.parse_args()

    eng = DMPEngine(args.ckpts, target_names=args.target_names)

    out_items, total, kept = [], 0, 0
    for it in tqdm(load_jsonl(args.in_jsonl), desc="Scoring"):
        total += 1
        smi = it.get("smiles") or it.get("SMILES") or it.get("mol", {}).get("smiles")
        if not smi:
            continue
        imps, top_atoms = per_atom_mask_importance(
            eng, smi, endpoint=args.endpoint, topk=args.topk, mode=args.mask_mode
        )
        if imps:
            it["importances"] = imps
            it["top_atoms"] = top_atoms
            kept += 1
        out_items.append(it)

    write_jsonl(out_items, args.out_jsonl)
    print(json.dumps({"read": total, "written": len(out_items), "with_importances": kept}, indent=2))

if __name__ == "__main__":
    main()
