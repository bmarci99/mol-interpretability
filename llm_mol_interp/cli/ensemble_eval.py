# llm_mol_interp/cli/ensemble_eval.py
from __future__ import annotations
import argparse, json, os, numpy as np, torch, pandas as pd
from torch_geometric.loader import DataLoader as GeoLoader
from llm_mol_interp.utils.dataset import Tox21Dataset
from llm_mol_interp.utils.splits import scaffold_split
from llm_mol_interp.utils.metrics import masked_auc, masked_aupr
from llm_mol_interp.models.dmpnn import DMPNN
from llm_mol_interp.utils.rdkit_logs import silence
silence(True)

def build_model_from_ckpt(ckpt_path: str, ex, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    h = ckpt.get("hparams", {})
    hidden = int(h.get("hidden_dim", 256))
    depth  = int(h.get("num_gnn_layers", 3))
    num_tasks = ckpt.get("num_tasks", None)
    m = DMPNN(atom_in_dim=ex.x.size(1),
              bond_in_dim=ex.edge_attr.size(1),
              hidden=hidden, depth=depth, dropout=0.2,
              num_tasks=num_tasks or 12).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--ckpts", nargs="+", required=True, help="paths to best.pt from each seed")
    ap.add_argument("--split-seed", type=int, default=0, help="scaffold split seed to define the eval split")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-preds-csv", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # dataset + split to evaluate on
    df = pd.read_csv(args.data_csv)
    task_cols = [c for c in df.columns if c != args.smiles_col]
    ds = Tox21Dataset(args.data_csv, args.smiles_col, task_cols)
    smiles = df[args.smiles_col].astype(str).tolist()
    _, _, te_idx = scaffold_split(smiles, (0.8, 0.1, 0.1), seed=args.split_seed)
    test  = [ds.get(i) for i in te_idx]
    test_loader = GeoLoader(test, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = test[0]
    models = [build_model_from_ckpt(p, ex, device) for p in args.ckpts]

    # inference
    Y_true, Y_prob = [], []
    ids = []  # (row idx) for optional CSV
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch.x = batch.x.float(); batch.edge_attr = batch.edge_attr.float(); batch.z = batch.z.long()
            logits_stack = []
            for m in models:
                lg, _, _ = m(batch.z, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                logits_stack.append(lg)
            avg_prob = torch.sigmoid(torch.stack(logits_stack, dim=0).mean(0)).cpu().numpy()
            y = batch.y
            if y.dim() == 1: y = y.view(avg_prob.shape[0], -1)
            Y_prob.append(avg_prob); Y_true.append(y.cpu().numpy())
            ids.extend(batch.idx.cpu().tolist() if hasattr(batch, "idx") else [None]*avg_prob.shape[0])

    Y_true = np.concatenate(Y_true, 0); Y_prob = np.concatenate(Y_prob, 0)
    auc, per = masked_auc(Y_true, Y_prob)
    aupr, _  = masked_aupr(Y_true, Y_prob)

    out = {
        "ensemble_size": len(models),
        "split_seed": args.split_seed,
        "test_macro_auc": None if np.isnan(auc) else float(auc),
        "test_macro_aupr": None if np.isnan(aupr) else float(aupr),
        "per_task_auc": [None if (isinstance(a,float) and np.isnan(a)) else float(a) for a in per],
        "ckpts": args.ckpts,
        "data_csv": args.data_csv,
        "smiles_col": args.smiles_col
    }
    with open(os.path.join(args.out_dir, "ensemble_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    #print(json.dumps(out, indent=2))

    if args.save_preds_csv:
        # optional per-molecule predictions (SMILES + per-task probs)
        te_df = df.iloc[te_idx].reset_index(drop=True).copy()
        prob_df = pd.DataFrame(Y_prob, columns=task_cols)
        out_df = pd.concat([te_df[[args.smiles_col]].reset_index(drop=True), prob_df], axis=1)
        out_df.to_csv(os.path.join(args.out_dir, "ensemble_test_predictions.csv"), index=False)

if __name__ == "__main__":
    main()
