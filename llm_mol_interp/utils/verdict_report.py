from __future__ import annotations
import argparse, json, csv, math, os, re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import yaml

# -----------------------
# Config / Heuristics
# -----------------------

# Simple, editable mapping from detected concept (alert) -> mitigation suggestions.
# Tweak/extend freely (or load from kb.yaml if you prefer).
MITIGATION_SUGGESTIONS = {
    # concept name in explanations.jsonl -> short actionable ideas
    "haloarene": [
        "Reduce halogenation or swap to less lipophilic substituents.",
        "Break planarity (e.g., add ortho substituent or sp3 linker).",
        "Introduce polarity/HBD/HBA to lower AhR binding and logP."
    ],
    "nitroaromatic": [
        "Remove/replace the nitro group; consider less redox-active EWG.",
        "Reduce aromatic activation (e.g., ring deactivation, meta placement).",
        "Add steric bulk to hinder metabolic activation."
    ],
    "anilide": [
        "Replace anilide with less reactive bioisostere.",
        "Block metabolic soft spots (e.g., ortho methyl)."
    ],
    "quinone_like": [
        "Avoid quinone/para-quinone motifs; reduce electrophilicity.",
        "Add electron-donating substituents or ring-break to reduce redox cycling."
    ],
    "Michael_acceptor": [
        "Remove α,β-unsaturated carbonyl or reduce conjugation.",
        "Add steric hindrance near the electrophile."
    ],
    "catechol": [
        "Block catechol oxidation (protecting groups or isosteres).",
        "Reduce planarity/conjugation; lower propensity for redox cycling."
    ],
}

def suggest_mitigation(concepts: List[str]) -> str:
    ideas: List[str] = []
    for c in concepts or []:
        c_key = c.strip().lower().replace(" ", "_")
        # normalize a few common variants
        c_key = c_key.replace("aryl_halide", "haloarene").replace("halo_arene","haloarene")
        if c_key in MITIGATION_SUGGESTIONS:
            ideas.extend(MITIGATION_SUGGESTIONS[c_key])
    # de-duplicate while preserving order
    seen = set()
    deduped = [x for x in ideas if not (x in seen or seen.add(x))]
    return " ".join(deduped[:3]) if deduped else ""

# -----------------------
# Thresholding
# -----------------------

def load_thresholds(thresholds_json: str | None, tasks: List[str], default_thr: float) -> Dict[str, float]:
    if thresholds_json and os.path.exists(thresholds_json):
        with open(thresholds_json, "r") as f:
            raw = json.load(f)
        return {t: float(raw.get(t, default_thr)) for t in tasks}
    return {t: default_thr for t in tasks}

def positive_tasks(probs: Dict[str, float], thr_map: Dict[str, float]) -> List[Tuple[str, float]]:
    pos = []
    for t, p in probs.items():
        thr = thr_map.get(t, 0.5)
        if (p is not None) and (not (isinstance(p, float) and math.isnan(p))) and p >= thr:
            pos.append((t, p))
    return sorted(pos, key=lambda kv: kv[1], reverse=True)

# -----------------------
# Main summarizer
# -----------------------

def summarize_explanations(
    explanations_jsonl: str,
    data_csv: str,
    smiles_col: str = "smiles",
    thresholds_json: str | None = None,
    default_thr: float = 0.5,
    out_csv: str | None = None,
    out_json: str | None = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:

    # Load JSONL
    records: List[Dict[str, Any]] = []
    with open(explanations_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Load labels
    df = pd.read_csv(data_csv)
    tasks = [c for c in df.columns if c != smiles_col]

    # Build a lookup from index_in_test -> true labels row
    # (explain_testset.py saves index_in_test derived from the test split indices)
    def get_true_labels(idx: int) -> Dict[str, Any]:
        row = df.iloc[idx]
        d = {t: (None if (pd.isna(row[t])) else int(row[t])) for t in tasks}
        d["_smiles"] = row[smiles_col]
        return d

    thr_map = load_thresholds(thresholds_json, tasks, default_thr)

    out_rows = []
    out_json_records = []

    for rec in records:
        probs: Dict[str, float] = rec.get("probs", {})
        concepts: List[str] = rec.get("detected_concepts", []) or []
        why = (rec.get("explanation") or "").replace("\n", " ").strip()
        idx = int(rec.get("index_in_test"))
        truelab = get_true_labels(idx)

        # verdict
        pos = positive_tasks(probs, thr_map)  # list of (task, prob) predicted positive
        toxic_tasks_str = ";".join([f"{t}:{p:.2f}" for t, p in pos])

        # top-3 for context
        top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_str = ";".join([f"{t}:{p:.2f}" for t, p in top3])

        # mitigation (heuristic)
        mitigation = suggest_mitigation(concepts)

        # true labels next to predictions
        # also compute a focus correctness flag (if true label available)
        focus_task = rec.get("focus_task")
        focus_prob = float(rec.get("focus_prob", np.nan))
        focus_true = truelab.get(focus_task, None)
        focus_correct = (None if focus_true is None else (1 if (focus_prob >= thr_map.get(focus_task, default_thr)) == bool(focus_true) else 0))

        # store CSV row
        row = {
            "index_in_test": idx,
            "smiles": rec.get("smiles"),
            "focus_task": focus_task,
            "focus_prob": round(focus_prob, 3) if not math.isnan(focus_prob) else "",
            "focus_true": focus_true,
            "focus_correct": focus_correct,
            "toxic_tasks_pred": toxic_tasks_str,
            "top3_tasks": top3_str,
            "detected_concepts": ";".join(concepts),
            "alert_overlap": ("" if rec.get("alert_overlap") is None else round(float(rec["alert_overlap"]), 3)),
            "why": why,
            "mitigation": mitigation,
            "image": rec.get("image", ""),
        }
        # add one column per task for (prob, true)
        for t in tasks:
            p = probs.get(t, None)
            row[f"{t}_prob"] = ("" if p is None else round(float(p), 3))
            tr = truelab.get(t, None)
            row[f"{t}_true"] = ("" if tr is None else int(tr))
        out_rows.append(row)

        # JSON record (nicely structured)
        out_json_records.append({
            "index_in_test": idx,
            "smiles": rec.get("smiles"),
            "focus": {
                "task": focus_task,
                "prob": focus_prob,
                "true": focus_true,
                "correct": focus_correct,
                "threshold": thr_map.get(focus_task, default_thr),
            },
            "predicted_positive_tasks": [{"task": t, "prob": p, "threshold": thr_map[t]} for t,p in pos],
            "top3_tasks": [{"task": t, "prob": p} for t,p in top3],
            "labels": {t: truelab.get(t, None) for t in tasks},
            "detected_concepts": concepts,
            "alert_overlap": rec.get("alert_overlap"),
            "why": why,
            "mitigation": mitigation,
            "image": rec.get("image"),
        })

    df_out = pd.DataFrame(out_rows).sort_values(["focus_task", "focus_prob"], ascending=[True, False])

    # Outputs
    if out_csv is None:
        out_csv = str(Path(explanations_jsonl).with_suffix("").as_posix() + "_verdicts.csv")
    if out_json is None:
        out_json = str(Path(explanations_jsonl).with_suffix("").as_posix() + "_verdicts.json")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    with open(out_json, "w") as f:
        json.dump(out_json_records, f, indent=2)

    return df_out, out_json_records

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Summarize explanations.jsonl into verdicts + why + true labels + mitigation.")
    ap.add_argument("--explanations-jsonl", required=True, help="Path to explanations.jsonl from explain_testset.py")
    ap.add_argument("--data-csv", required=True, help="Original data CSV with true labels")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--thresholds-json", default=None, help="Optional JSON {task: threshold} to override 0.5")
    ap.add_argument("--default-thr", type=float, default=0.5)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df_out, _ = summarize_explanations(
        args.explanations_jsonl,
        args.data_csv,
        smiles_col=args.smiles_col,
        thresholds_json=args.thresholds_json,
        default_thr=args.default_thr,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )
    print(f"[OK] wrote {args.out_csv or Path(args.explanations_jsonl).with_suffix('').as_posix() + '_verdicts.csv'}")
    print(f"Rows: {len(df_out)} | Columns: {len(df_out.columns)}")

if __name__ == "__main__":
    main()
