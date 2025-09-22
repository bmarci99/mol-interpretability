import argparse, os, numpy as np, torch
from rdkit import Chem

from llm_mol_interp.models.dmpnn import DMPNN
from llm_mol_interp.utils.featurizer import mol_to_pyg
from llm_mol_interp.xai.attribution import integrated_gradients_edge_importance
from llm_mol_interp.xai.substructures import select_alert_by_overlap
from llm_mol_interp.xai.visualize import draw_mol_with_highlight
from llm_mol_interp.xai.rag_explainer import GroundedExplainer

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--gnn-checkpoint", required=True)
    ap.add_argument("--task", default=None, help="optional task to focus (e.g., SR-p53)")
    ap.add_argument("--out", default=None, help="output directory (defaults next to checkpoint)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load checkpoint & reconstruct D-MPNN
    ckpt = torch.load(args.gnn_checkpoint, map_location=device)
    sd  = ckpt.get("model", ckpt.get("state_dict", ckpt))
    num_tasks = ckpt.get("num_tasks", len(TASKS))
    hp = ckpt.get("hparams", {}) or {}
    tmp = mol_to_pyg(Chem.MolFromSmiles("c1ccccc1"))
    node_in_dim, edge_in_dim = tmp.x.size(1), tmp.edge_attr.size(1)

    model = DMPNN(node_in_dim, edge_in_dim,
                  hidden=hp.get("hidden_dim", 256),
                  depth=hp.get("num_gnn_layers", 3),
                  num_tasks=num_tasks).to(device)
    model.load_state_dict(sd, strict=True); model.eval()

    # ---- build graph for the input molecule
    mol = Chem.MolFromSmiles(args.smiles)
    data = mol_to_pyg(mol); data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data = data.to(device)

    # ---- predict
    with torch.no_grad():
        logits, _, _ = model(data.z, data.node_feats, data.edge_index, data.edge_attr, data.batch)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    pred = {TASKS[i]: float(probs[i]) for i in range(len(probs))}
    print("Predictions (top 5):")
    for k in sorted(pred, key=pred.get, reverse=True)[:5]:
        print(f"  {k:12s} p={pred[k]:.3f}")

    # ---- choose focus task (argmax unless user picked one)
    if args.task:
        focus = TASKS.index(args.task)
    else:
        focus = int(np.argmax(probs))

    # ---- attribution → substructure detection
    bonds = integrated_gradients_edge_importance(model, data, task_idx=focus, steps=64)  # [(i,j,importance)]
    alert = select_alert_by_overlap(mol, bonds, thr=0.6, mode="hybrid")
    if alert:
        faithful_tag = "faithful" if alert.get("faithful", False) else "presence-only"
        print(f"Detected key feature: {alert['name']} (score={alert['score']:.2f}, {faithful_tag})")
        atom_ids = alert["atom_indices"]
    else:
        print("No structural alert matched.")
        atom_ids = []



    # ---- outputs
    out_dir = args.out or (os.path.dirname(args.gnn_checkpoint) or ".")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "explanation.png")
    draw_mol_with_highlight(args.smiles, atom_ids=atom_ids, bond_pairs=bonds, out_path=png_path)
    print(f"Saved visualization to {png_path}")

    # ---- grounded explanation text
    #explainer = GroundedExplainer()  # loads kb.yaml from package data
    #text = explainer.generate(alert["name"] if alert else "none", pred)

    # ---- grounded explanation text
    from llm_mol_interp.xai.rag_explainer import GroundedExplainer
    expl = GroundedExplainer(kb_path="llm_mol_interp/xai/kb.yaml")

    # prepare detected concepts list
    detected_concepts = [alert["name"]] if alert else []
    proba = pred  # reuse your predictions dict

    single_text = expl.generate(
        sub_name=detected_concepts[0] if detected_concepts else None,
        pred_dict=proba
    )

    multi_text = expl.generate_multi(
        detected_concepts, proba, max_sentences=2
    )

    print("Explanation (single):", single_text)
    print("Explanation (multi) :", multi_text)
    text = multi_text if multi_text else single_text

    # save a small JSON alongside
    import json
    with open(os.path.join(out_dir, "explanation.json"), "w") as f:
        json.dump({
            "smiles": args.smiles,
            "focus_task": TASKS[focus],
            "focus_prob": float(probs[focus]),
            "detected_alert": (alert["name"] if alert else None),
            "overlap": (float(alert["score"]) if alert else None),
            "explanation": text,
            "png": os.path.abspath(png_path),
            "probs": pred
        }, f, indent=2)
    print(f"Saved JSON to {os.path.join(out_dir, 'explanation.json')}")

if __name__ == "__main__":
    main()
