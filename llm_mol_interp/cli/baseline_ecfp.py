import argparse, numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def smi2ecfp(smi, nBits=2048, r=2):
    m = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=nBits)
    arr = np.zeros((1,), dtype=np.int8); DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    args = ap.parse_args()
    df = pd.read_csv(args.data_csv)
    tasks = [c for c in df.columns if c!=args.smiles_col]
    X = np.vstack([smi2ecfp(s) for s in df[args.smiles_col]])
    for t in tasks:
        y = df[t].values
        mask = ~np.isnan(y)
        Xtr, Xte, ytr, yte = train_test_split(X[mask], y[mask], test_size=0.2, random_state=0, stratify=y[mask])
        clf = XGBClassifier(n_estimators=300, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, p)
        print(f"{t:12s} AUC={auc:.3f}")

if __name__ == "__main__":
    main()
