from __future__ import annotations
import json, time, os, typer, torch
from pathlib import Path
from rdkit import Chem
from typing import List, Tuple
from typing_extensions import Annotated

from llm_mol_interp.models.dmpnn import DMPNN
from llm_mol_interp.utils.featurizer import mol_to_pyg
from llm_mol_interp.xai.attribution import integrated_gradients_edge_importance
from llm_mol_interp.xai.substructures import select_alert_by_overlap
from llm_mol_interp.xai.visualize import draw_mol_with_highlight
import torch.nn.functional as F
# Optional LLM (prompt-only or LoRA). If you don't want LLM, we do rule-text.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

app = typer.Typer(add_completion=False)

def predict(model, data, task_idx: int) -> float:
    with torch.no_grad():
        batch = data.batch
        if batch is None:
            batch = torch.zeros(data.z.size(0), dtype=torch.long, device=data.z.device)
        logits, _, _ = model(data.z, data.node_feats, data.edge_index, data.edge_attr, batch)
        return float(torch.sigmoid(logits[0, task_idx]).item())

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"]

    # Infer dims directly from weights
    embed_dim = sd["embed_z.weight"].shape[1]                 # e.g., 32
    in_atom  = sd["atom_in.weight"].shape[1]                  # e.g., 43 (= 32 + F_train)
    F_train  = in_atom - embed_dim                            # e.g., 11
    E_train  = sd["bond_in.weight"].shape[1]                  # bond feature dim

    hidden = ckpt.get("hparams",{}).get("hidden_dim", 256)
    depth  = ckpt.get("hparams",{}).get("num_gnn_layers", 3)
    num_tasks = ckpt["num_tasks"]

    model = DMPNN(F_train, E_train, hidden=hidden, depth=depth, num_tasks=num_tasks).to(device).eval()
    model.load_state_dict(sd)
    tasks = ckpt.get("tasks", None)
    # Return expected dims so we can check/pad inputs
    return model, tasks, F_train, E_train



def rule_text(task, pred_label, p, sub_name):
    def article(s): return "an" if s and s[0].lower() in "aeiou" else "a"
    if sub_name:
        if pred_label == "toxic":
            return (f"Prediction: {pred_label} (p={p:.2f}) for {task}. "
                    f"The model focused on {article(sub_name)} {sub_name}, a motif often associated with risk in some endpoints.")
        else:
            return (f"Prediction: {pred_label} (p={p:.2f}) for {task}. "
                    f"The model was most sensitive around {article(sub_name)} {sub_name}; "
                    f"however, its overall evidence for {task} indicates low risk.")
    return (f"Prediction: {pred_label} (p={p:.2f}) for {task}. "
            f"No single alert dominated; the decision appears distributed.")

def maybe_llm_text(mode: str, llm_dir: str|None, task: str, pred_label: str, p: float, sub_name: str|None) -> str:
    if mode == "rule" or sub_name is None:
        return rule_text(task, pred_label, p, sub_name)

    tok = AutoTokenizer.from_pretrained(llm_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(llm_dir)

    subs = sub_name or "none"
    prompt = (
        f"Task: {task}\n"
        f"Model output: {pred_label} (p={p:.2f}).\n"
        f"Model sensitivity region(s): {subs}.\n"
        "Write 1–2 neutral sentences that:\n"
        " - Refer explicitly to THIS task only.\n"
        " - Describe where the model focused.\n"
        " - Avoid claiming that the task itself is a substructure.\n"
        " - Avoid over-generalizing toxicophore rules.\n"
        "Explanation:"
    )

    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(**ids, max_new_tokens=96, no_repeat_ngram_size=4, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def propose_edits(mol, alert_name, alert_smarts, predictor, task_idx, device, F_train, E_train) -> List[
        Tuple[str, float, str]]:

    from rdkit.Chem import AllChem
    import torch.nn.functional as F

    def _replace(smiles_repl: str):
        patt = Chem.MolFromSmarts(alert_smarts)
        repl = Chem.MolFromSmiles(smiles_repl)
        outs = Chem.ReplaceSubstructs(mol, patt, repl, replaceAll=True)
        return outs[0] if outs else None

    candidates = []
    # Generic edits per alert (very heuristic; you can refine later)
    recipes = []
    if "nitro" in alert_name:
        recipes = [("replace nitro with hydrogen", "[H]"),
                   ("replace nitro with fluorine", "F"),
                   ("replace nitro with methyl", "C")]
    elif "Michael acceptor" in alert_name or "enone" in alert_name:
        recipes = [("saturate the double bond (approx.)", "CCC(=O)C"),
                   ("mask with alcohol (approx.)", "CC(O)C(=O)C")]
    elif "epoxide" in alert_name:
        recipes = [("open epoxide to diol (approx.)", "OCCO")]
    elif "nitrosamine" in alert_name:
        recipes = [("replace nitrosamine with amide", "NC=O"),
                   ("replace nitrosamine with tertiary amide", "N(C)C=O")]
    elif "quinone" in alert_name:
        recipes = [("reduce quinone (approx.)", "Oc1cccc(O)c1")]
    elif "anilide" in alert_name:
        recipes = [("move acyl away from aniline (approx.)", "Nc1ccccc1"),
                   ("replace anilide with aliphatic amide", "NC(=O)C")]

    # Always include plain removal as last resort
    recipes.append(("remove alert motif", "[H]"))

    for desc, repl in recipes:
        m2 = _replace(repl)
        if m2 is None:
            continue
        try:
            Chem.SanitizeMol(m2)
            smi2 = Chem.MolToSmiles(m2)
            d2 = mol_to_pyg(m2).to(device)

            # --- pad to training widths
            if d2.node_feats.size(1) < F_train:
                d2.node_feats = F.pad(d2.node_feats, (0, F_train - d2.node_feats.size(1)))
            elif d2.node_feats.size(1) > F_train:
                d2.node_feats = d2.node_feats[:, :F_train]

            if d2.edge_attr.size(1) < E_train:
                d2.edge_attr = F.pad(d2.edge_attr, (0, E_train - d2.edge_attr.size(1)))
            elif d2.edge_attr.size(1) > E_train:
                d2.edge_attr = d2.edge_attr[:, :E_train]

            if getattr(d2, "batch", None) is None:
                import torch
                d2.batch = torch.zeros(d2.z.size(0), dtype=torch.long, device=device)

            p2 = predict(predictor, d2, task_idx)
            candidates.append((desc, p2, smi2))
        except Exception:
            continue

    return sorted(candidates, key=lambda x: x[1])

@app.command()
def ask(
    smiles: str = typer.Argument(..., help="SMILES string of the molecule"),
    ckpt: str = typer.Option("runs/dmpnn_scaffold256x3/best.pt"),
    task: str = typer.Option("SR-MMP"),
    lora_dir: str = typer.Option(""),
    out_dir: str = typer.Option("runs/ask"),
    device_str: Annotated[str, typer.Option("--device", help="cpu|cuda|mps")] = "cpu",
    task_idx_opt: Annotated[int | None, typer.Option("--task-idx", help="Override model head index")] = None,  # <-- MOVE HERE
):
    os.makedirs(out_dir, exist_ok=True)

    #device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    # pick device safely
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")



    #model, tasks = load_model(ckpt, device)
    model, tasks, F_train, E_train = load_model(ckpt, device)

    if task_idx_opt is not None:
        task_idx = task_idx_opt
    elif tasks and task in tasks:
        task_idx = tasks.index(task)
    else:
        task_idx = 0  # temporary fallback if mapping fails (see Option 2 below)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        typer.echo("Invalid SMILES.")
        raise typer.Exit(1)

    data = mol_to_pyg(mol).to(device)
    #import torch.nn.functional as F

    # Ensure atom features match checkpoint width (F_train)
    Fa = data.node_feats.size(1)
    if Fa < F_train:
        data.node_feats = F.pad(data.node_feats, (0, F_train - Fa), value=0.0)
    elif Fa > F_train:
        data.node_feats = data.node_feats[:, :F_train]

    # Ensure bond features match checkpoint width (E_train)
    Fe = data.edge_attr.size(1)
    if Fe < E_train:
        data.edge_attr = F.pad(data.edge_attr, (0, E_train - Fe), value=0.0)
    elif Fe > E_train:
        data.edge_attr = data.edge_attr[:, :E_train]
    # NEW: ensure batch exists (single graph => all zeros)
    if getattr(data, "batch", None) is None:
        data.batch = torch.zeros(data.z.size(0), dtype=torch.long, device=device)

    p = predict(model, data, task_idx)
    pred_label = "toxic" if p >= 0.5 else "non-toxic"

    # Attribution → select alert
    bonds = integrated_gradients_edge_importance(model, data, task_idx=task_idx, steps=32)
    sel = select_alert_by_overlap(mol, [(i,j,w) for (i,j,w) in bonds], thr=0.4)
    alert_name = sel["name"] if sel else None
    alert_smarts = sel.get("smarts","") if sel else ""


    #####NEW
    from collections import defaultdict
    atom_score = defaultdict(float)
    for i, j, w in bonds:
        s = abs(float(w))
        atom_score[int(i)] += s
        atom_score[int(j)] += s

    K = max(3, int(0.2 * len(atom_score)))
    top_atoms = [a for a, _ in sorted(atom_score.items(), key=lambda kv: kv[1], reverse=True)[:K]]

    if sel and "atom_indices" in sel:
        atom_ids = list(set(sel["atom_indices"]) | set(top_atoms))
    else:
        atom_ids = top_atoms

    # Explanation
    mode = "rule" if not lora_dir else "lora"
    text = maybe_llm_text(mode, lora_dir or None, task, pred_label, p, sel["name"] if sel else None)

    # Visualization (single tag)
    tag = time.strftime("%Y%m%d_%H%M%S")
    png_path = f"{out_dir}/card_{tag}.png"
    draw_mol_with_highlight(
        Chem.MolToSmiles(mol),
        atom_ids=atom_ids,  # ← keep the widened set
        bond_pairs=bonds,
        out_path=png_path,
        size=(600, 450),
        signed=True           # if your visualize supports it
    )

    # Counterfactuals (with padding)
    proposals = []
    if sel:
        proposals = propose_edits(mol, alert_name, alert_smarts, model, task_idx, device, F_train, E_train)[:5]

    # Save JSON result
    out = {
        "smiles": smiles,
        "task": task,
        "probability": p,
        "prediction": pred_label,
        "alert": sel or {},
        "explanation": text,
        "png": png_path,
        "suggestions": [{"edit": d, "new_prob": float(pp), "smiles": s} for (d, pp, s) in proposals],
    }
    out_json = f"{out_dir}/card_{tag}.json"
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    # Console summary
    typer.echo(f"\nTask: {task}\nProb: {p:.3f} → {pred_label}")
    if alert_name:
        typer.echo(f"Highlighted: {alert_name}")
    typer.echo(f"Explanation: {text}")
    typer.echo(f"Figure: {png_path}")
    if proposals:
        typer.echo("\nTop suggestions (lower prob is better):")
        for (desc, p2, smi2) in proposals[:3]:
            typer.echo(f" - {desc}: p={p2:.3f}   {smi2}")
    typer.echo(f"\nSaved JSON: {out_json}")

if __name__ == "__main__":
    app()