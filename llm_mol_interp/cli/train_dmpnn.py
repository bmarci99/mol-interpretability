# llm_mol_interp/cli/train_dmpnn.py
from __future__ import annotations
import argparse, os, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange
from torch_geometric.loader import DataLoader as GeoLoader

from llm_mol_interp.utils.dataset import Tox21Dataset
from llm_mol_interp.utils.splits import scaffold_split
from llm_mol_interp.utils.metrics import masked_auc, masked_aupr
from llm_mol_interp.utils.rdkit_logs import silence
from llm_mol_interp.models.dmpnn import DMPNN

# Silence RDKit warnings early (presentable logs)
silence(warnings=True)

def seed_everything(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def compute_pos_weights(Y_train_np: np.ndarray) -> torch.Tensor:
    """
    Y_train_np: [N,T] with NaN/-1 for missing. Computes per-task neg/pos ratio on TRAIN ONLY.
    """
    Y = Y_train_np.copy()
    Y[~np.isfinite(Y)] = -1
    pw = []
    T = Y.shape[1]
    for t in range(T):
        m = Y[:, t] >= 0
        if m.sum() == 0:
            pw.append(1.0); continue
        pos = (Y[m, t] == 1).sum()
        neg = (Y[m, t] == 0).sum()
        pw.append(float(neg / max(pos, 1)))
    return torch.tensor(pw, dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--scheduler", choices=["none","cosine","onecycle"], default="onecycle")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    # --- load data & split
    df = pd.read_csv(args.data_csv)
    task_cols = [c for c in df.columns if c != args.smiles_col]
    print(f"[INFO] Detected {len(task_cols)} tasks: {task_cols}")

    ds = Tox21Dataset(args.data_csv, args.smiles_col, task_cols)  # expects batch.x(float), batch.z(long), batch.y(float with NaN)
    smiles = df[args.smiles_col].astype(str).tolist()
    tr_idx, va_idx, te_idx = scaffold_split(smiles, (0.8, 0.1, 0.1), seed=args.seed)

    train = [ds.get(i) for i in tr_idx]
    valid = [ds.get(i) for i in va_idx]
    test  = [ds.get(i) for i in te_idx]
    print(f"[INFO] split sizes: train={len(train)}  val={len(valid)}  test={len(test)}")

    train_loader = GeoLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoLoader(valid, batch_size=256)
    test_loader  = GeoLoader(test,  batch_size=256)

    num_tasks = len(task_cols)
    ex = train[0]

    # --- model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMPNN(atom_in_dim=ex.x.size(1),
                  bond_in_dim=ex.edge_attr.size(1),
                  hidden=args.hidden, depth=args.depth,
                  dropout=args.dropout, num_tasks=num_tasks
                  ).to(device)

    # --- pos_weight from TRAIN ONLY
    Y_tr = df.loc[tr_idx, task_cols].to_numpy().astype(np.float32)
    pos_weight = compute_pos_weights(Y_tr).to(device)  # shape [T]
    # Build BCE once with pos_weight
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    # --- optimizer / scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1, len(train_loader))
        )
    elif args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = None

    best_auc, best_path = -1.0, os.path.join(args.out_dir, "best.pt")
    pat_cnt = 0

    # --- build BCE once (with pos_weight) ---
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    for ep in trange(1, args.epochs + 1):
        # ---- train
        model.train();
        tr_losses = []
        first_batch_debug = True
        for batch in train_loader:
            batch = batch.to(device)
            # enforce correct dtypes
            batch.x = batch.x.float()
            batch.edge_attr = batch.edge_attr.float()
            batch.z = batch.z.long()

            logits, _, _ = model(batch.z, batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [B,T]
            y = batch.y
            if y.dim() == 1:
                y = y.view(logits.size(0), -1)

            # mask missing labels
            #mask = (~torch.isnan(y)).float()  # 1 if label exists
            #y_filled = torch.nan_to_num(y, nan=0.0)  # replace NaN with 0 for BCE input

            # mask missing labels: use y >= 0 (since your dataset uses -1 for missing)
            mask = (y >= 0).float()
            y_filled = torch.where(y < 0, torch.zeros_like(y), y)  # replace -1 with 0 for BCE input

            # elementwise BCE (already includes pos_weight)
            loss_elem = bce(logits, y_filled)  # [B,T]  (>=0)
            valid = mask.sum()
            if valid == 0:
                continue
            loss = (loss_elem * mask).sum() / valid  # scalar (>=0)

            # --- safety net: assert non-negative, finite ---
            li = float(loss.detach().cpu())
            if not np.isfinite(li) or li < 0:
                print("[DEBUG] Bad loss detected.")
                print("  loss:", li,
                      "min(loss_elem):", float(loss_elem.min().detach().cpu()),
                      "max(loss_elem):", float(loss_elem.max().detach().cpu()),
                      "valid_count:", float(valid.detach().cpu()))
                print("  logits stats:", float(logits.min().detach().cpu()), float(logits.max().detach().cpu()))
                print("  y unique:", torch.unique(y[~torch.isnan(y)]).tolist())
                raise RuntimeError("Loss is non-finite or negative.")

            opt.zero_grad();
            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            if args.scheduler == "onecycle":
                sched.step()
            tr_losses.append(li)

            # one-time wiring debug on first batch
            if first_batch_debug:
                first_batch_debug = False
                print(f"[DBG] x {tuple(batch.x.shape)} dtype={batch.x.dtype}; "
                      f"z {tuple(batch.z.shape)} dtype={batch.z.dtype}; "
                      f"edge_index {tuple(batch.edge_index.shape)}; "
                      f"edge_attr {tuple(batch.edge_attr.shape)} dtype={batch.edge_attr.dtype}; "
                      f"y has NaN={(torch.isnan(y).any().item())}")
        if args.scheduler == "cosine":
            sched.step()

        # ---- validation (AUC for early stopping)
        model.eval();
        Y_true, Y_prob = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                batch.x = batch.x.float();
                batch.edge_attr = batch.edge_attr.float();
                batch.z = batch.z.long()
                logits, _, _ = model(batch.z, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                prob = torch.sigmoid(logits).cpu().numpy()
                y = batch.y
                if y.dim() == 1:
                    y = y.view(logits.size(0), -1)
                Y_prob.append(prob);
                Y_true.append(y.cpu().numpy())

        Y_true = np.concatenate(Y_true, 0) if len(Y_true) else np.zeros((0, num_tasks))
        Y_prob = np.concatenate(Y_prob, 0) if len(Y_prob) else np.zeros((0, num_tasks))
        val_auc, per_auc = masked_auc(Y_true, Y_prob)
        val_aupr, _ = masked_aupr(Y_true, Y_prob)
        tr_loss = (np.mean(tr_losses) if tr_losses else float("nan"))
        print(
            f"Epoch {ep:03d} | train_loss {tr_loss:.4f} | val AUC {val_auc:.4f} | val AUPR {val_aupr:.4f} | best {best_auc:.4f}")

        improved = (not np.isnan(val_auc)) and ((val_auc - best_auc) > args.min_delta or best_auc < 0)
        if improved:
            best_auc = val_auc;
            pat_cnt = 0
            torch.save({
                "model": model.state_dict(),
                "num_tasks": num_tasks,
                "hparams": {"hidden_dim": args.hidden, "num_gnn_layers": args.depth}
            }, best_path)
            with open(os.path.join(args.out_dir, "val_metrics.json"), "w") as f:
                json.dump({
                    "val_macro_auc": float(val_auc),
                    "val_macro_aupr": (None if np.isnan(val_aupr) else float(val_aupr)),
                    "per_task_auc": [None if (isinstance(a, float) and np.isnan(a)) else float(a) for a in per_auc]
                }, f)
            print(f"[saved best] {best_path}")
        else:
            pat_cnt += 1
            if pat_cnt >= args.patience:
                print(f"[INFO] Early stopping at epoch {ep} (no ΔAUC ≥ {args.min_delta} for {args.patience} epochs)")
                break

    # ---- test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    Y_true, Y_prob = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch.x = batch.x.float()
            batch.edge_attr = batch.edge_attr.float()
            batch.z = batch.z.long()
            logits, _, _ = model(batch.z, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            y = batch.y
            if y.dim() == 1: y = y.view(logits.size(0), -1)
            Y_prob.append(prob); Y_true.append(y.cpu().numpy())

    if len(Y_true) == 0:
        macro_auc, per = float('nan'), []
    else:
        Y_true = np.concatenate(Y_true, 0); Y_prob = np.concatenate(Y_prob, 0)
        macro_auc, per = masked_auc(Y_true, Y_prob)

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump({"test_macro_auc": (None if np.isnan(macro_auc) else float(macro_auc)),"per_task_auc": [None if (isinstance(a, float) and np.isnan(a)) else float(a) for a in per]}, f)
    print(f"[DONE] Best val AUC {best_auc:.4f} | Test AUC {macro_auc:.4f}")
    #print(f"Saved best checkpoint: {best_path}")

    # ---- consolidated metrics.json (thesis-friendly)
    conf = {
        "data_csv": args.data_csv,
        "split": "scaffold 80/10/10",
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "batch_size": args.batch_size,
        "seed": args.seed
    }
    out = {
        "best_val_macro_auc": float(best_auc),
        "test_macro_auc": (None if np.isnan(macro_auc) else float(macro_auc)),
        "val_metrics_json": os.path.join(args.out_dir, "val_metrics.json"),
        "test_metrics_json": os.path.join(args.out_dir, "test_metrics.json"),
        "checkpoint": best_path,
        "config": conf,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {os.path.join(args.out_dir, 'metrics.json')}")

if __name__ == "__main__":
    main()
