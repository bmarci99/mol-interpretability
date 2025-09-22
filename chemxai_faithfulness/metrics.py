# Lightweight faithfulness evaluation toolkit for molecular explainers.
# Implements: Comprehensiveness, Sufficiency, Deletion/Insertion AUCs, and KB-driven Counterfactual impact.
# Expects a predictor wrapper that can output per-endpoint probabilities from SMILES.
# Author: You | License: MIT

import json, os, argparse, statistics
from typing import List, Dict, Any, Optional, Tuple

try:
    from rdkit import Chem
except Exception:
    Chem = None

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def moving_auc(xs, ys):
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        area += dx * (ys[i] + ys[i-1]) / 2.0
    return area

class EnsemblePredictor:
    def __init__(self, ckpt_paths: List[str], target_names: Optional[List[str]]=None):
        self.ckpt_paths = ckpt_paths
        self.target_names = target_names or []
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            from llm_mol_interp.runtime.predictor import DMPEngine
            self._model = DMPEngine(self.ckpt_paths, self.target_names)
        except Exception as e:
            raise RuntimeError("Implement llm_mol_interp.runtime.predictor.DMPEngine") from e

    def predict_proba(self, smiles: str) -> Dict[str, float]:
        self._lazy_init()
        return self._model.predict_proba(smiles)

    def predict_proba_with_mask(self, smiles: str, mask_atoms: List[int], mode: str="zero") -> Dict[str, float]:
        self._lazy_init()
        if hasattr(self._model, "predict_proba_with_mask"):
            return self._model.predict_proba_with_mask(smiles, mask_atoms, mode=mode)
        # fallback: hard delete via RDKit
        if Chem is None:
            raise RuntimeError("RDKit not available; provide predict_proba_with_mask in your DMPEngine")
        edited = smiles_delete_atoms(smiles, mask_atoms)
        return self._model.predict_proba(edited)

def smiles_delete_atoms(smiles: str, atom_indices: List[int]) -> str:
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    emol = Chem.RWMol(mol)
    for idx in sorted(set(atom_indices), reverse=True):
        if 0 <= idx < emol.GetNumAtoms():
            emol.RemoveAtom(idx)
    new = emol.GetMol()
    try:
        Chem.SanitizeMol(new)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(new, canonical=True)
    except Exception:
        return smiles

def smiles_keep_only_atoms(smiles: str, atom_indices: List[int]) -> str:
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    keep = set(atom_indices)
    amap = {}
    em = Chem.EditableMol(Chem.Mol())
    for i, a in enumerate(mol.GetAtoms()):
        if i in keep:
            amap[i] = em.AddAtom(Chem.Atom(a.GetAtomicNum()))
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a in keep and c in keep:
            em.AddBond(amap[a], amap[c], b.GetBondType())
    new = em.GetMol()
    try:
        Chem.SanitizeMol(new)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(new, canonical=True)
    except Exception:
        return smiles

def comprehensiveness(p0: float, p_masked: float) -> float:
    return max(0.0, p0 - p_masked)

def sufficiency(p_only: float, p0: float) -> float:
    return max(0.0, p_only)

def deletion_insertion_curve(predict_fn, smiles: str, importance: List[float], topk: Optional[int]=None) -> Dict[str, float]:
    n = len(importance)
    order = sorted(range(n), key=lambda i: importance[i], reverse=True)
    topk = min(topk or n, n)

    xs_del, ys_del = [0.0], []
    ys_del.append(predict_fn(smiles, [], mode="zero"))
    masked = []
    for t in range(1, topk+1):
        masked.append(order[t-1])
        p = predict_fn(smiles, masked, mode="zero")
        xs_del.append(t/topk)
        ys_del.append(p)
    auc_del = moving_auc(xs_del, ys_del)

    xs_ins, ys_ins = [0.0], []
    masked = list(range(n))
    ys_ins.append(predict_fn(smiles, masked, mode="zero"))
    for t in range(1, topk+1):
        masked.remove(order[t-1])
        p = predict_fn(smiles, masked, mode="zero")
        xs_ins.append(t/topk)
        ys_ins.append(p)
    auc_ins = moving_auc(xs_ins, ys_ins)
    return {"deletion_auc": auc_del, "insertion_auc": auc_ins}

def apply_kb_edit(smiles: str, kb_edit: Dict[str, Any]) -> Optional[str]:
    if Chem is None:
        return None
    if kb_edit.get("strategy", "smarts_replace") != "smarts_replace":
        return None
    patt = kb_edit.get("smarts", None)
    repl = kb_edit.get("replacement_smiles", "")
    if not patt:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    patt_mol = Chem.MolFromSmarts(patt)
    if patt_mol is None:
        return None
    matches = mol.GetSubstructMatches(patt_mol)
    if not matches:
        return None
    idxs = sorted(set(matches[0]), reverse=True)
    em = Chem.RWMol(mol)
    for i in idxs:
        if 0 <= i < em.GetNumAtoms():
            em.RemoveAtom(i)
    new = em.GetMol()
    try:
        Chem.SanitizeMol(new)
    except Exception:
        pass
    if repl:
        try:
            frag = Chem.MolFromSmiles(repl)
            if frag is not None:
                combo = Chem.CombineMols(new, frag)
                Chem.SanitizeMol(combo)
                new = combo
        except Exception:
            pass
    try:
        return Chem.MolToSmiles(new, canonical=True)
    except Exception:
        return None

def parse_important_atoms(item: Dict[str, Any]) -> Tuple[List[int], Optional[List[float]], List[str]]:
    atoms, scores, kb_ids = [], None, []

    # 1) Canonical schema
    if "top_substructures" in item:
        for s in item["top_substructures"]:
            if isinstance(s, dict):
                if "atom_indices" in s and isinstance(s["atom_indices"], list):
                    atoms += [int(a) for a in s["atom_indices"]]
                if "kb_id" in s:
                    kb_ids.append(s["kb_id"])
        atoms = sorted(set(atoms))
    elif "top_atoms" in item and isinstance(item["top_atoms"], list):
        atoms = [int(a) for a in item["top_atoms"]]

    # 2) Common alternates from various pipelines
    # per-atom score arrays
    for k in ["importances", "ig_scores", "atom_scores", "saliency", "attribution", "ig_atom_importances"]:
        if k in item and isinstance(item[k], list):
            scores = [float(v) for v in item[k]]
            break

    # top-k atoms under different names
    for k in ["ig_topk_atoms", "highlight_atoms", "salient_atoms", "atoms"]:
        if not atoms and k in item and isinstance(item[k], list):
            atoms = sorted(set(int(a) for a in item[k]))
            break

    # nested containers (e.g., {"explanation": {"atoms": [...], "kb_id": ...}})
    for k in ["explanation", "explanations", "xai", "rationale"]:
        if k in item:
            sub = item[k]
            if isinstance(sub, dict):
                if not atoms and isinstance(sub.get("atoms"), list):
                    atoms = sorted(set(int(a) for a in sub["atoms"]))
                if "kb_id" in sub:
                    kb_ids.append(sub["kb_id"])
                if scores is None and isinstance(sub.get("scores"), list):
                    scores = [float(v) for v in sub["scores"]]
            elif isinstance(sub, list):
                for s in sub:
                    if isinstance(s, dict):
                        if "atoms" in s and isinstance(s["atoms"], list):
                            atoms += [int(a) for a in s["atoms"]]
                        if "kb_id" in s:
                            kb_ids.append(s["kb_id"])
                        if scores is None and isinstance(s.get("scores"), list):
                            scores = [float(v) for v in s["scores"]]
                atoms = sorted(set(atoms))

    # last resort: packed string of indices "3,4,5"
    if not atoms:
        for k in ["atoms_str", "highlight_str"]:
            if k in item and isinstance(item[k], str):
                try:
                    atoms = sorted(set(int(a.strip()) for a in item[k].split(",") if a.strip().isdigit()))
                    break
                except Exception:
                    pass

    return atoms, scores, kb_ids


def eval_item(predictor: EnsemblePredictor, smiles: str, endpoint: str,
              important_atoms: List[int], importances: Optional[List[float]],
              kb: Dict[str, Any], kb_ids_for_item: List[str]) -> Dict[str, Any]:
    p_full = predictor.predict_proba(smiles)[endpoint]
    p_drop = predictor.predict_proba_with_mask(smiles, important_atoms, mode="zero")[endpoint]
    comp = comprehensiveness(p_full, p_drop)

    # Sufficiency
    if importances and hasattr(predictor._model, "predict_proba_with_mask"):
        n = len(importances)
        complement = [i for i in range(n) if i not in set(important_atoms)]
        p_only = predictor.predict_proba_with_mask(smiles, complement, mode="zero")[endpoint]
    else:
        kept = smiles_keep_only_atoms(smiles, important_atoms)
        p_only = predictor.predict_proba(kept)[endpoint] if kept else 0.0
    suff = sufficiency(p_only, p_full)

    # Deletion/Insertion
    del_ins = {}
    if importances and hasattr(predictor._model, "predict_proba_with_mask"):
        def pf(_sm, mask_atoms, mode="zero"):
            return predictor.predict_proba_with_mask(_sm, mask_atoms, mode=mode)[endpoint]
        del_ins = deletion_insertion_curve(pf, smiles, importances, topk=min(len(importances), 50))

    # Counterfactuals
    cf_results = []
    concepts = {c.get("id"): c for c in kb.get("concepts", [])}
    for kid in kb_ids_for_item:
        c = concepts.get(kid)
        if not c: continue
        for edit in c.get("counterfactual_edits", []):
            new_smiles = apply_kb_edit(smiles, edit)
            if not new_smiles: continue
            p_new = predictor.predict_proba(new_smiles)[endpoint]
            cf_results.append({
                "kb_id": kid,
                "edit": edit.get("name"),
                "original_p": p_full,
                "edited_p": p_new,
                "delta": p_new - p_full,
                "edited_smiles": new_smiles
            })

    out = {
        "endpoint": endpoint,
        "p_full": p_full,
        "p_drop": p_drop,
        "comprehensiveness": comp,
        "p_only": p_only,
        "sufficiency": suff,
        "counterfactuals": cf_results
    }
    out.update(del_ins)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations-jsonl", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kb", default=None)
    ap.add_argument("--target-names", nargs="*", default=None)
    args = ap.parse_args()

    items = load_jsonl(args.explanations_jsonl)

    kb = {}
    if args.kb:
        import yaml
        with open(args.kb, "r", encoding="utf-8") as f:
            kb = yaml.safe_load(f)

    predictor = EnsemblePredictor(args.ckpts, args.target_names)

    results = []
    for it in items:
        smiles = it.get("smiles") or it.get("SMILES") or it.get("mol", {}).get("smiles")
        if not smiles: continue
        atoms, scores, kb_ids = parse_important_atoms(it)
        kb_ids = kb_ids or []
        res = eval_item(predictor, smiles, args.endpoint, atoms, scores, kb, kb_ids)
        results.append({"smiles": smiles, **res})

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"faithfulness_{args.endpoint}.json")
    save_json(results, out_json)

    summary = {
        "N": len(results),
        "comprehensiveness_mean": statistics.fmean(r["comprehensiveness"] for r in results) if results else None,
        "sufficiency_mean": statistics.fmean(r["sufficiency"] for r in results) if results else None,
        "deletion_auc_mean": statistics.fmean(r.get("deletion_auc", 0.0) for r in results) if results else None,
        "insertion_auc_mean": statistics.fmean(r.get("insertion_auc", 0.0) for r in results) if results else None,
        "counterfactual_any_decrease_frac": (
            sum(1 for r in results if any(cf["delta"] < 0 for cf in r.get("counterfactuals", [])))/len(results)
            if results else None
        )
    }
    out_sum = os.path.join(args.out_dir, f"faithfulness_{args.endpoint}_summary.json")
    save_json(summary, out_sum)
    print(json.dumps(summary, indent=2))
