#!/usr/bin/env python3
"""
Score predictions with a given thresholds_flat05.json.

Usage:
  python utils/score_with_thresholds.py \
    --jsonl runs/.../explanations.jsonl \
    --csv data/tox21_multitask.csv \
    --smiles-col smiles \
    --thresholds runs/.../thresholds_f1.json
"""
import argparse, json, numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--thresholds", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.jsonl)]
    df   = pd.read_csv(args.csv)
    tasks = [c for c in df.columns if c != args.smiles_col]
    thr   = json.load(open(args.thresholds))

    def labels(i):
        row = df.iloc[i]
        return {t: (None if pd.isna(row[t]) else int(row[t])) for t in tasks}

    stats = {t: {"tp":0,"fp":0,"tn":0,"fn":0} for t in tasks}
    for r in recs:
        y = labels(int(r["index_in_test"]))
        p = r["probs"]
        for t in tasks:
            if y[t] is None or p.get(t) is None:
                continue
            yhat = int(float(p[t]) >= thr.get(t,0.5)); yt = y[t]
            if yhat and yt: stats[t]["tp"]+=1
            elif yhat and not yt: stats[t]["fp"]+=1
            elif (not yhat) and (not yt): stats[t]["tn"]+=1
            elif (not yhat) and yt: stats[t]["fn"]+=1

    rows=[]
    for t,s in stats.items():
        n = sum(s.values())
        if n==0: continue
        prec = s["tp"]/max(1,(s["tp"]+s["fp"]))
        rec  = s["tp"]/max(1,(s["tp"]+s["fn"]))
        f1   = 0 if prec+rec==0 else 2*prec*rec/(prec+rec)
        acc  = (s["tp"]+s["tn"])/n
        rows.append([t,n,s["tp"],s["fp"],s["tn"],s["fn"],round(prec,3),round(rec,3),round(f1,3),round(acc,3),thr.get(t,0.5)])
    df_out = pd.DataFrame(rows, columns=["task","n","tp","fp","tn","fn","precision","recall","f1","accuracy","threshold"]).sort_values("f1", ascending=False)
    print(df_out)

    # macro/micro
    macro_f1 = df_out["f1"].mean()
    tp=sum(s["tp"] for s in stats.values()); fp=sum(s["fp"] for s in stats.values())
    tn=sum(s["tn"] for s in stats.values()); fn=sum(s["fn"] for s in stats.values())
    micro_prec = tp/max(1,tp+fp); micro_rec = tp/max(1,tp+fn)
    micro_f1 = 0 if micro_prec+micro_rec==0 else 2*micro_prec*micro_rec/(micro_prec+micro_rec)
    micro_acc = (tp+tn)/max(1,tp+fp+tn+fn)
    print(f"\nMacro F1: {macro_f1:.3f} | Micro F1: {micro_f1:.3f} | Micro Acc: {micro_acc:.3f}")

if __name__ == "__main__":
    main()
