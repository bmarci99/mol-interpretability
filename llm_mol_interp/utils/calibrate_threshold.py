#!/usr/bin/env python3
"""
Calibrate per-task decision thresholds.

Outputs:
  - thresholds_f1.json   : per-task cutoff maximizing F1
  - thresholds_p050.json : per-task cutoff meeting precision >= 0.50 if possible, else F1-optimal

Usage:
  python utils/calibrate_thresholds.py \
    --jsonl runs/.../explanations.jsonl \
    --csv data/tox21_multitask.csv \
    --smiles-col smiles \
    --target-precision 0.50 \
    --out-dir runs/.../
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def collect_pairs(jsonl, csv, smiles_col):
    recs = [json.loads(l) for l in open(jsonl)]
    df = pd.read_csv(csv)
    tasks = [c for c in df.columns if c != smiles_col]
    pairs = {t: {"p": [], "y": []} for t in tasks}
    for r in recs:
        idx = int(r["index_in_test"])
        probs = r["probs"]
        row = df.iloc[idx]
        for t in tasks:
            y = row[t]
            p = probs.get(t, None)
            if p is None or pd.isna(y):
                continue
            pairs[t]["p"].append(float(p))
            pairs[t]["y"].append(int(y))
    return tasks, pairs

def best_thr_f1(ps, ys):
    if not ps: return 0.5
    ps, ys = np.array(ps), np.array(ys)
    best_g, best_f1 = 0.5, -1.0
    for g in np.linspace(0.05, 0.95, 181):
        yhat = ps >= g
        tp = ((yhat==1)&(ys==1)).sum(); fp = ((yhat==1)&(ys==0)).sum(); fn = ((yhat==0)&(ys==1)).sum()
        prec = tp / max(1, tp+fp); rec = tp / max(1, tp+fn)
        f1 = 0 if prec+rec==0 else 2*prec*rec/(prec+rec)
        if f1 > best_f1: best_f1, best_g = f1, float(g)
    return best_g

def pick_thr_precision(ps, ys, target_prec, fallback_f1):
    if not ps: return fallback_f1
    ps, ys = np.array(ps), np.array(ys)
    best_g, best_prec = None, -1.0
    for g in np.linspace(0.05, 0.95, 181):
        yhat = ps >= g
        tp = ((yhat==1)&(ys==1)).sum(); fp = ((yhat==1)&(ys==0)).sum()
        prec = tp / max(1, tp+fp)
        if prec >= target_prec and prec > best_prec:
            best_g, best_prec = float(g), float(prec)
    return best_g if best_g is not None else fallback_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--target-precision", type=float, default=0.50)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tasks, pairs = collect_pairs(args.jsonl, args.csv, args.smiles_col)

    thr_f1 = {t: best_thr_f1(pairs[t]["p"], pairs[t]["y"]) for t in tasks}
    thr_p  = {t: pick_thr_precision(pairs[t]["p"], pairs[t]["y"], args.target_precision, thr_f1[t]) for t in tasks}

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"thresholds_f1.json").write_text(json.dumps(thr_f1, indent=2))
    (out/"thresholds_p050.json").write_text(json.dumps(thr_p, indent=2))
    print("Wrote", out/"thresholds_f1.json")
    print("Wrote", out/"thresholds_p050.json")

if __name__ == "__main__":
    main()
