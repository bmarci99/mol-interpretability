import argparse, numpy as np, pandas as pd, torch, inspect
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from rdkit import Chem

from llm_mol_interp.utils.dataset import Tox21Dataset
from llm_mol_interp.utils.splits import scaffold_split
from llm_mol_interp.utils.featurizer import mol_to_pyg
from llm_mol_interp.models.dmpnn import DMPNN
import os

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
NR = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma']
SR = ['SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

def _looks_like_gat(sd):   return any(k.startswith("gats.") or k.endswith("att_src") for k in sd)
def _looks_like_dmpnn(sd): return any(k.startswith("atom_in") or k.startswith("bond_in") or "W_msg" in k for k in sd)

def _infer_gat_hparams_from_state(sd):
    if "out.weight" in sd: hidden_dim = sd["out.weight"].shape[1]
    elif "proj_in.weight" in sd: hidden_dim = sd["proj_in.weight"].shape[0]
    else: hidden_dim = 128
    layers = [int(k.split(".")[1]) for k in sd if k.startswith("gats.") and k.split(".")[1].isdigit()]
    num_layers = (max(layers)+1) if layers else 3
    att_keys = [k for k in sd if k.endswith("att_src")]
    if att_keys:
        _, num_heads, head_dim = sd[att_keys[0]].shape  # [1,H,D]
    else:
        num_heads, head_dim = 4, max(1, hidden_dim//4)
    return {"hidden_dim": hidden_dim, "num_gnn_layers": num_layers, "num_heads": num_heads, "head_dim": head_dim}

def _import_gat_predictor():
    for modpath in ("llm_mol_interp.models.gat",
                    "llm_mol_interp.models.gat_predictor",
                    "llm_mol_interp.models.gnn_gat"):
        try:
            mod = __import__(modpath, fromlist=["GATPredictor"])
            return getattr(mod, "GATPredictor")
        except Exception:
            pass
    raise ImportError("Could not import GATPredictor. Update _import_gat_predictor() to your actual path.")

def _instantiate_gat_with_matching_args(GATPredictor, node_in_dim, edge_in_dim, num_tasks, h, device):
    sig = inspect.signature(GATPredictor.__init__)
    params = set(sig.parameters)
    def put(names, val, kw):
        for n in names:
            if n in params: kw[n] = val; return
    kw = {}
    put(["node_in_dim","in_dim","node_dim","num_node_features"], node_in_dim, kw)
    put(["edge_in_dim","edge_dim","num_edge_features"], edge_in_dim, kw)
    put(["hidden_dim","hidden","hidden_channels","emb_dim","embed_dim"], h["hidden_dim"], kw)
    put(["num_gnn_layers","num_layers","depth","n_layers"], h["num_gnn_layers"], kw)
    put(["num_heads","heads","n_heads"], h["num_heads"], kw)
    put(["head_dim","att_dim","attention_dim","per_head_dim"], h["head_dim"], kw)
    put(["num_tasks","out_dim","out_channels","n_tasks"], num_tasks, kw)
    model = GATPredictor(**kw).to(device)
    print(f"[GAT] hidden={h['hidden_dim']} layers={h['num_gnn_layers']} heads={h['num_heads']} head_dim={h['head_dim']}")
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--gnn-checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    task_cols = [c for c in df.columns if c != args.smiles_col]
    ds = Tox21Dataset(args.data_csv, args.smiles_col, task_cols)

    tr, va, te = scaffold_split(df[args.smiles_col].tolist(), (0.8,0.1,0.1), seed=args.seed)
    test_list = [ds.get(i) for i in te]
    loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.gnn_checkpoint, map_location=device)
    sd  = ckpt.get("model", ckpt.get("state_dict", ckpt))
    num_tasks = ckpt.get("num_tasks", len(task_cols))
    hparams   = ckpt.get("hparams", None)

    tmp = mol_to_pyg(Chem.MolFromSmiles("c1ccccc1"))
    node_in_dim, edge_in_dim = tmp.x.size(1), tmp.edge_attr.size(1)

    # --- auto-select architecture ---
    if _looks_like_gat(sd):
        if _looks_like_gat(sd):
            try:
                GATPredictor = _import_gat_predictor()  # your helper
            except ImportError:
                raise SystemExit(
                    "This checkpoint is GAT, but no GATPredictor is present in your codebase.\n"
                    "→ Either pass a D-MPNN checkpoint (--gnn-checkpoint <path>) or add your GATPredictor class."
                )
            # (if you DO have the class, instantiate as before)
        #GATPredictor = _import_gat_predictor()
        h = hparams or _infer_gat_hparams_from_state(sd)
        model = _instantiate_gat_with_matching_args(GATPredictor, node_in_dim, edge_in_dim, num_tasks, h, device)
    else:
        hidden = (hparams or {}).get("hidden_dim", 256)
        depth  = (hparams or {}).get("num_gnn_layers", 3)
        model = DMPNN(node_in_dim, edge_in_dim, hidden=hidden, depth=depth, num_tasks=num_tasks).to(device)
        print(f"[DMPNN] hidden={hidden} depth={depth}")

    model.load_state_dict(sd, strict=True)
    model.eval()

    Y_true, Y_prob = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits,_,_ = model(b.z, b.node_feats, b.edge_index, b.edge_attr, b.batch)
            Y_prob.append(torch.sigmoid(logits).cpu().numpy())
            Y_true.append(b.y.cpu().numpy())
    Y_true = np.concatenate(Y_true,0); Y_prob = np.concatenate(Y_prob,0)

    aucs = {}
    for i, t in enumerate(task_cols):
        m = Y_true[:,i] >= 0
        if m.sum() < 5 or len(np.unique(Y_true[m,i])) < 2:
            aucs[t] = np.nan
        else:
            aucs[t] = roc_auc_score(Y_true[m,i], Y_prob[m,i])

    ser = pd.Series(aucs).rename("AUC")
    ser.loc["NR"]  = ser[NR].mean(skipna=True)
    ser.loc["SR"]  = ser[SR].mean(skipna=True)
    ser.loc["AVG"] = ser[TASKS].mean(skipna=True)

    print("\nPer-task ROC-AUC (test, scaffold split):")
    for t in TASKS:
        print(f"{t:12s} {ser.loc[t]:.3f}")
    print(f"\nNR mean: {ser.loc['NR']:.3f}")
    print(f"SR mean: {ser.loc['SR']:.3f}")
    print(f"Macro AVG: {ser.loc['AVG']:.3f}")

    ser.to_csv("runs/per_task_auc.csv")

if __name__ == "__main__":
    main()
