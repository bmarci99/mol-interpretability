from __future__ import annotations
import argparse, os, json, numpy as np, torch, pandas as pd
from torch_geometric.loader import DataLoader as GeoLoader
from rdkit import Chem
import torch

from llm_mol_interp.utils.dataset import Tox21Dataset
from llm_mol_interp.utils.splits import scaffold_split
from llm_mol_interp.utils.rdkit_logs import silence
from llm_mol_interp.models.dmpnn import DMPNN

from llm_mol_interp.xai.attribution import integrated_gradients_edge_importance
from llm_mol_interp.xai.substructures import select_alert_by_overlap
from llm_mol_interp.xai.visualize import draw_mol_with_highlight
from llm_mol_interp.xai.rag_explainer import GroundedExplainer

silence(True)

def load_model(ckpt_path, fallback_atom_in_dim, bond_in_dim, num_tasks, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt["model"]

    # first-layer input width from checkpoint
    W_in = sd["atom_in.weight"].shape[1]

    # detect the embedding: your model uses 'embed_z'
    emb_key = None
    if "embed_z.weight" in sd:
        emb_key = "embed_z.weight"
    elif "atom_emb.weight" in sd:
        emb_key = "atom_emb.weight"

    if emb_key is not None:
        embed_dim = sd[emb_key].shape[1]
        node_feats_dim = W_in - embed_dim   # since atom_in sees [embed || node_feats]
        has_emb = True
    else:
        # fallback (rare): no embedding param saved -> assume atom_in consumes only node feats
        embed_dim = 0
        node_feats_dim = W_in
        has_emb = False

    h = ckpt.get("hparams", {})
    hidden = int(h.get("hidden_dim", 256))
    depth  = int(h.get("num_gnn_layers", 3))
    n_tasks = int(ckpt.get("num_tasks", num_tasks))

    # Your DMPNN always has embed_z and expects to concat 32-dim embedding.
    # So here we must ensure 'embed_dim' equals what the architecture uses (32).
    # If the checkpoint had embed_z, embed_dim should already be 32.
    # Build the model with node_feats_dim; DMPNN will add its 32-dim embed internally.
    m = DMPNN(
        atom_in_dim=node_feats_dim,     # node feature width (NO Z)
        bond_in_dim=bond_in_dim,
        hidden=hidden, depth=depth, dropout=0.2,
        num_tasks=n_tasks
    ).to(device)

    m.load_state_dict(sd)  # strict load, should now match
    m.eval()

    print(f"[debug] ckpt atom_in={W_in}, emb_key={emb_key}, emb_dim={embed_dim}, node_feats_dim={node_feats_dim}")
    return m, node_feats_dim



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--ckpt", required=True, help="model used for attribution")
    ap.add_argument("--ensemble-ckpts", nargs="*", default=None, help="optional: use these checkpoints to ensemble probabilities")
    ap.add_argument("--kb", default="llm_mol_interp/xai/kb.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-mols", type=int, default=100, help="limit for quick runs; -1 for all")
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--template-only", action="store_true", help="don’t load HF model; use grounded template with citations only")
    ap.add_argument("--thresholds-json", default=None)
    ap.add_argument("--default-thr", type=float, default=0.5)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    thr = {}
    if args.thresholds_json and os.path.exists(args.thresholds_json):
        with open(args.thresholds_json) as f: thr = json.load(f)
    # --- data & split (scaffold)
    df = pd.read_csv(args.data_csv)

    def true_labels_for_row(idx: int) -> dict:
        """Return {task: 0/1 or None} for a given CSV row index."""
        row = df.iloc[idx]
        out = {}
        for t in task_cols:
            val = row.get(t, None)
            if pd.isna(val):
                out[t] = None
            else:
                out[t] = int(val)
        return out

    # placeholders for single-SMILES (no label columns)
    DEFAULT_TOX21 = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
                     "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
                     "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

    dataset_path = args.data_csv
    task_cols = [c for c in df.columns if c != args.smiles_col]

    if len(task_cols) == 0:
        task_cols = DEFAULT_TOX21
        print("[info] No label columns found; using placeholders:", task_cols)
        # add NaN columns so Tox21Dataset can read them
        for t in task_cols:
            if t not in df.columns:
                df[t] = np.nan
        tmp_csv = os.path.join(args.out_dir, "_tmp_single_unlabeled.csv")
        df.to_csv(tmp_csv, index=False)
        dataset_path = tmp_csv

    # build dataset with the *right path*
    ds = Tox21Dataset(dataset_path, args.smiles_col, task_cols)

    smiles_all = df[args.smiles_col].astype(str).tolist()
    tr_idx, va_idx, te_idx = scaffold_split(smiles_all, (0.8, 0.1, 0.1), seed=args.split_seed)
    test_idxs = te_idx if args.max_mols < 0 else te_idx[:args.max_mols]
    test_smiles = [smiles_all[i] for i in test_idxs]
    test_datas = [ds.get(i) for i in test_idxs]
    loader = GeoLoader(test_datas, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # infer dims from first example
    ex = test_datas[0]
    #atom_in_dim = ex.x.size(1)
    #bond_in_dim = ex.edge_attr.size(1)
    atom_in_dim = ex.node_feats.size(1)  # IMPORTANT: node features only (no Z)
    bond_in_dim = ex.edge_attr.size(1)

    if len(task_cols) == 0:
        ckpt_meta = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        n_tasks_expected = int(ckpt_meta.get("num_tasks", 12))
        task_cols = [f"task_{i}" for i in range(n_tasks_expected)]
        print(f"[info] No label columns found; using {n_tasks_expected} placeholder task names:", task_cols)

    num_tasks = len(task_cols)

    # --- attribution model
    #model = load_model(args.ckpt, atom_in_dim, bond_in_dim, num_tasks, device)
    model, expected_node_feats = load_model(args.ckpt, atom_in_dim, bond_in_dim, num_tasks, device)

    ckpt_dbg = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    expected_atom_in = ckpt_dbg["model"]["atom_in.weight"].shape[1]
    #print("[debug] ckpt expects atom_in.in_features =", expected_atom_in)
    #print("[debug] dataset node_feats dim =", atom_in_dim)  # from ex.x.size(1) vs node_feats, depending on your DMPNN

    # --- optional ensemble for probabilities
    ensemble_models = []
    ensemble_dims = []
    if args.ensemble_ckpts and len(args.ensemble_ckpts) > 0:
        for p in args.ensemble_ckpts:
            m_i, dim_i = load_model(p, atom_in_dim, bond_in_dim, num_tasks, device)
            ensemble_models.append(m_i)
            ensemble_dims.append(dim_i)

        # (optional) assert dims match across seeds
        if len(set(ensemble_dims)) != 1:
            print("[warn] ensemble node feature dims differ:", ensemble_dims)

    # --- explainer (LLM or template)
    expl = GroundedExplainer(kb_path=args.kb, model_name=None if args.template_only else "google/flan-t5-base")

    out_jsonl = os.path.join(args.out_dir, "explanations.jsonl")
    with open(out_jsonl, "w") as fout:
        for n, (batch, smi) in enumerate(zip(loader, test_smiles)):
            # inside the loop, after dtype casts
            batch = batch.to(device)
            batch.x = batch.x.float();
            batch.edge_attr = batch.edge_attr.float();
            batch.z = batch.z.long()
            labels = true_labels_for_row(int(test_idxs[n]))

            # build the correct node feature tensor
            x_node = batch.node_feats if hasattr(batch, "node_feats") else batch.x[:, 1:]

            # Decision info (thresholds optional)
            #fname = task_cols[focus] if 0 <= focus < len(task_cols) else f"task_{focus}"

            #cut = float(thr.get(fname, args.default_thr))
            #focus_pred = 1 if probs[focus] >= cut else 0
            #focus_true = labels.get(fname, None)
            #focus_correct = (None if focus_true is None else int(focus_pred == focus_true))

            # width guard against ckpt expectation
            cur = x_node.size(1)
            exp = expected_node_feats  # returned by load_model(...)
            if cur < exp:
                pad = exp - cur
                x_node = torch.cat([x_node, torch.zeros(x_node.size(0), pad, device=x_node.device, dtype=x_node.dtype)],
                                   dim=1)
            elif cur > exp:
                x_node = x_node[:, :exp]

            # ensure IG sees the right features too
            batch.node_feats = x_node

            # forward pass to get probabilities
            with torch.no_grad():
                if ensemble_models:
                    logits = []
                    for m in ensemble_models:
                        lg, _, _ = m(batch.z, x_node, batch.edge_index, batch.edge_attr, batch.batch)
                        logits.append(lg)
                    probs = torch.sigmoid(torch.stack(logits, 0).mean(0)).cpu().numpy()[0]
                else:
                    lg, _, _ = model(batch.z, x_node, batch.edge_index, batch.edge_attr, batch.batch)
                    probs = torch.sigmoid(lg).cpu().numpy()[0]

            proba = {task_cols[i]: float(probs[i]) for i in range(num_tasks)}
            focus = int(np.nanargmax(probs))

            # Attribution for the focus task
            bonds = integrated_gradients_edge_importance(model, batch, task_idx=focus, steps=args.ig_steps)

            # RDKit mol for mapping & viz
            mol = Chem.MolFromSmiles(smi)

            # Select structural alert by overlap with attributions
            alert = select_alert_by_overlap(mol, bonds, thr=0.6, mode="hybrid")
            if alert:
                detected_concepts = [alert["name"]]
                atom_ids = alert.get("atom_indices", [])
            else:
                detected_concepts, atom_ids = [], []

            # Visualization (optional)
            png_path = os.path.join(args.out_dir, f"exp_{n:04d}.png")
            try:
                draw_mol_with_highlight(smi, atom_ids=atom_ids, bond_pairs=bonds, out_path=png_path)
            except Exception:
                png_path = None

            # Text generation
            if detected_concepts:
                text = expl.generate_multi(detected_concepts, proba, max_sentences=2)
            else:
                text = expl.generate(None, proba)

            # --- Decision info (now that we have probs + focus)
            fname = task_cols[focus] if 0 <= focus < len(task_cols) else f"task_{focus}"
            cut = float(thr.get(fname, args.default_thr))
            focus_pred = 1 if probs[focus] >= cut else 0
            focus_true = labels.get(fname, None)
            focus_correct = (None if focus_true is None else int(focus_pred == focus_true))

            rec = {
                "index_in_test": int(test_idxs[n]),
                "smiles": smi,
                "focus_task": fname,
                "focus_prob": float(probs[focus]),
                "probs": {task_cols[i]: float(probs[i]) for i in range(num_tasks)},
                "detected_concepts": detected_concepts,
                "alert_overlap": (None if not alert else float(alert.get("score", 0.0))),
                "explanation": text,
                "image": png_path,
                "labels": labels,
                "focus_true": focus_true,
                "focus_threshold": cut,
                "focus_pred_label": focus_pred,
                "focus_correct": focus_correct
            }
            fout.write(json.dumps(rec) + "\n")
            print(f"[OK] {n:04d} {smi} | concept={detected_concepts} | {fname}={probs[focus]:.3f}")


if __name__ == "__main__":
    main()
