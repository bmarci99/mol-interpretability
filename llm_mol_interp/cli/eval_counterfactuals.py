import argparse, numpy as np, torch
from rdkit import Chem
from rdkit.Chem import AllChem
from llm_mol_interp.models.dmpnn import DMPNN
from llm_mol_interp.utils.featurizer import mol_to_pyg
from llm_mol_interp.xai.substructures import SMARTS
from llm_mol_interp.xai.attribution import integrated_gradients_edge_importance
from llm_mol_interp.xai.substructures import select_alert_by_overlap

def remove_alert(mol, smarts, replacement="[H]"):
    patt = Chem.MolFromSmarts(smarts)
    repl = Chem.MolFromSmiles(replacement)
    out = Chem.ReplaceSubstructs(mol, patt, repl, replaceAll=True)
    if not out: return None
    m = out[0]
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m

def predict(model, mol):
    data = mol_to_pyg(mol); data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data = data.to(next(model.parameters()).device)
    with torch.no_grad():
        p = torch.sigmoid(model(data.z, data.node_feats, data.edge_index, data.edge_attr, data.batch)[0]).cpu().numpy()[0]
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--gnn-checkpoint", required=True)
    ap.add_argument("--alert", default="aromatic nitro")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp = mol_to_pyg(Chem.MolFromSmiles("c1ccccc1"))
    ckpt = torch.load(args.gnn_checkpoint, map_location=device)
    model = DMPNN(tmp.x.size(1), tmp.edge_attr.size(1), hidden=256, depth=3, num_tasks=ckpt["num_tasks"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    mol = Chem.MolFromSmiles(args.smiles)
    p0 = predict(model, mol)
    m_edit = remove_alert(mol, SMARTS[args.alert], replacement="[H]")
    if m_edit is None:
        print("Could not create counterfactual molecule.")
        return
    p1 = predict(model, m_edit)
    delta = p0 - p1
    print("Δ prob (original - edited) per task:", np.round(delta, 3))
    print("Mean Δ:", float(delta.mean()))
if __name__ == "__main__":
    main()
