# llm_mol_interp/cli/train_baseline_ecfp.py
from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from llm_mol_interp.utils.splits import scaffold_split
from llm_mol_interp.baselines.ecfp import smiles_to_ecfp_bits

def masked_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    """Compute macro AUC/AUPR across tasks, ignoring NaNs in y_true."""
    T = y_true.shape[1]
    aucs, auprs = [], []
    for t in range(T):
        yt = y_true[:, t]
        yp = y_prob[:, t]
        mask = np.isfinite(yt)
        yt = yt[mask]; yp = yp[mask]
        if len(np.unique(yt)) < 2:
            aucs.append(np.nan); auprs.append(np.nan); continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except Exception:
            aucs.append(np.nan)
        try:
            auprs.append(average_precision_score(yt, yp))
        except Exception:
            auprs.append(np.nan)
    per_auc = np.array(aucs, dtype=float)
    per_aupr = np.array(auprs, dtype=float)
    macro_auc = (np.nan if np.all(np.isnan(per_auc)) else float(np.nanmean(per_auc)))
    macro_aupr = (np.nan if np.all(np.isnan(per_aupr)) else float(np.nanmean(per_aupr)))
    return macro_auc, macro_aupr, per_auc, per_aupr

def build_clf(name: str, seed: int, scale_pos_weight: float | None):
    if name == "xgb" and _HAS_XGB:
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            reg_lambda=1.0, reg_alpha=0.0, random_state=seed,
            n_jobs=0, scale_pos_weight=scale_pos_weight
        )
    # fallback: liblinear LR
    return LogisticRegression(
        solver="liblinear", penalty="l2", max_iter=2000, class_weight="balanced",
        random_state=seed
    )

def main():
    ap = argparse.ArgumentParser(description="Train an ECFP baseline (per-task XGB or LogisticRegression) with scaffold split.")
    ap.add_argument("--data-csv", required=True, help="CSV with SMILES + binary task columns")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--clf", choices=["xgb", "logreg"], default="xgb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-preds-csv", action="store_true")
    ap.add_argument("--save-jsonl", action="store_true", help="Write test predictions JSONL for threshold calibration")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"models").mkdir(exist_ok=True)

    df = pd.read_csv(args.data_csv)
    smiles = df[args.smiles_col].astype(str).tolist()
    task_cols = [c for c in df.columns if c != args.smiles_col]
    T = len(task_cols)
    print(f"[INFO] Detected {T} tasks: {task_cols}")

    # Global scaffold split (shared across tasks)
    tr_idx, va_idx, te_idx = scaffold_split(smiles, (0.8, 0.1, 0.1), seed=args.seed,
                                            min_val=100, min_test=100)

    # Compute fingerprints once
    X = smiles_to_ecfp_bits(smiles, n_bits=args.n_bits, radius=args.radius)

    # Train per-task
    models = {}
    test_probs = np.full((len(smiles), T), np.nan, dtype=float)
    val_probs  = np.full((len(smiles), T), np.nan, dtype=float)

    for ti, t in enumerate(task_cols):
        y_all = df[t].to_numpy()
        # Build per-split masks with valid labels
        tr_mask = np.isfinite(y_all) & np.isin(np.arange(len(smiles)), tr_idx)
        va_mask = np.isfinite(y_all) & np.isin(np.arange(len(smiles)), va_idx)
        te_mask = np.isfinite(y_all) & np.isin(np.arange(len(smiles)), te_idx)

        n_pos = int((df.loc[tr_mask, t] == 1).sum())
        n_neg = int((df.loc[tr_mask, t] == 0).sum())
        spw = None
        if n_pos > 0:
            spw = max(1.0, n_neg / max(1, n_pos))

        clf = build_clf(args.clf, args.seed, spw)
        Xtr, ytr = X[tr_mask], y_all[tr_mask].astype(int)
        Xva, yva = X[va_mask], y_all[va_mask].astype(int)
        Xte, yte = X[te_mask], y_all[te_mask].astype(int)

        if Xtr.shape[0] == 0 or len(np.unique(ytr)) < 2:
            print(f"[WARN] Task {t}: not enough labeled training data; skipping.")
            continue

        if args.clf == "xgb" and _HAS_XGB:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            clf.fit(Xtr, ytr)

        # Save model per task (joblib if available)
        try:
            import joblib
            joblib.dump(clf, out / "models" / f"{t}.joblib")
            models[t] = str(out / "models" / f"{t}.joblib")
        except Exception:
            models[t] = "<not saved>"

        # Collect probs on val/test for metrics
        if hasattr(clf, "predict_proba"):
            pva = clf.predict_proba(Xva)[:, 1]
            pte = clf.predict_proba(Xte)[:, 1]
        else:
            # LR decision function fallback
            pva = 1.0 / (1.0 + np.exp(-clf.decision_function(Xva)))
            pte = 1.0 / (1.0 + np.exp(-clf.decision_function(Xte)))

        # write into global arrays at original indices
        val_probs[va_mask, ti] = pva
        test_probs[te_mask, ti] = pte

        # Per-task metrics
        try:
            auc = roc_auc_score(yte, pte) if len(np.unique(yte))==2 else np.nan
        except Exception:
            auc = np.nan
        try:
            aupr = average_precision_score(yte, pte) if len(np.unique(yte))==2 else np.nan
        except Exception:
            aupr = np.nan
        print(f"[OK] {t:12s} | val n={Xva.shape[0]:4d} | test n={Xte.shape[0]:4d} | AUC={auc if isinstance(auc,float) else np.nan:.3f} | AUPR={aupr if isinstance(aupr,float) else np.nan:.3f}")

    # Macro metrics
    Y = df[task_cols].to_numpy()
    _, _, per_auc, per_aupr = masked_metrics(Y[te_idx], test_probs[te_idx])
    macro_auc = (np.nan if np.all(np.isnan(per_auc)) else float(np.nanmean(per_auc)))
    macro_aupr = (np.nan if np.all(np.isnan(per_aupr)) else float(np.nanmean(per_aupr)))

    # Save metrics + config
    out_json = {
        "test_macro_auc": None if np.isnan(macro_auc) else float(macro_auc),
        "test_macro_aupr": None if np.isnan(macro_aupr) else float(macro_aupr),
        "per_task_auc": [None if (isinstance(a,float) and np.isnan(a)) else float(a) for a in per_auc],
        "per_task_aupr": [None if (isinstance(a,float) and np.isnan(a)) else float(a) for a in per_aupr],
        "task_names": task_cols,
        "split_indices": {"train": tr_idx, "val": va_idx, "test": te_idx},
        "models": models,
        "config": {
            "n_bits": args.n_bits, "radius": args.radius, "clf": ("xgb" if (_HAS_XGB and args.clf=="xgb") else "logreg"),
            "seed": args.seed
        },
        "data_csv": args.data_csv, "smiles_col": args.smiles_col
    }
    (out/"metrics.json").write_text(json.dumps(out_json, indent=2))
    print(f"Wrote {(out/'metrics.json')}")

    # Optional: save predictions CSV on test set
    if args.save_preds_csv:
        te_df = df.iloc[te_idx].reset_index(drop=True).copy()
        prob_df = pd.DataFrame(test_probs[te_idx], columns=task_cols)
        out_df = pd.concat([te_df[[args.smiles_col]].reset_index(drop=True), prob_df], axis=1)
        out_df.to_csv(out/"baseline_test_predictions.csv", index=False)
        print(f"Wrote {(out/'baseline_test_predictions.csv')}")

    # Optional: JSONL for threshold calibration (compatible with utils/calibrate_threshold.py)
    if args.save_jsonl:
        jsonl_path = out/"baseline_test_predictions.jsonl"
        with open(jsonl_path, "w") as f:
            for local_i, global_i in enumerate(te_idx):
                rec = {
                    "index_in_test": int(global_i),
                    "smiles": smiles[global_i],
                    "probs": {task_cols[j]: (None if np.isnan(test_probs[global_i, j]) else float(test_probs[global_i, j]))
                              for j in range(T)}
                }
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {jsonl_path}")

if __name__ == "__main__":
    main()