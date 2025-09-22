import torch
from captum.attr import IntegratedGradients

class LogitForTask(torch.nn.Module):
    def __init__(self, model, task_idx):
        super().__init__()
        self.model, self.t = model, task_idx

    def forward(self, z, x, edge_index, edge_attr, batch):
        # Return one logit PER example: [B]
        logits, _, _ = self.model(z, x, edge_index, edge_attr, batch)  # [B, T]
        return logits[:, self.t]  # [B]

def integrated_gradients_edge_importance(
    model, data, task_idx: int, steps: int = 64, internal_bs: int = 0, signed: bool = True
):
    """
    Integrated gradients over edge_attr (bonds).
    Returns [(i, j, importance)] for undirected bonds with SIGNED weights if signed=True
    (positive => increases tox logit, negative => decreases).
    """
    device = next(model.parameters()).device
    model.eval()

    # Ensure batch exists (single-graph input => all zeros)
    if getattr(data, "batch", None) is None:
        data.batch = torch.zeros(data.z.size(0), dtype=torch.long, device=device)

    edge_attr = data.edge_attr.detach().to(device).requires_grad_(True)
    baseline  = torch.zeros_like(edge_attr)

    wrapper = LogitForTask(model, task_idx)

    def f(edge_attr_in):
        # Captum may pass [E, d] or [S, E, d]; sum over steps if present
        if edge_attr_in.dim() == 3:
            total = None
            for s in range(edge_attr_in.size(0)):
                out = wrapper(data.z, data.node_feats, data.edge_index, edge_attr_in[s], data.batch)  # [B]
                total = out if total is None else (total + out)
            return total  # [B]
        else:
            return wrapper(data.z, data.node_feats, data.edge_index, edge_attr_in, data.batch)        # [B]

    ig = IntegratedGradients(f)
    if internal_bs <= 0:
        internal_bs = int(edge_attr.size(0))  # safe for small molecules

    attr = ig.attribute(edge_attr, baselines=baseline, n_steps=steps, internal_batch_size=internal_bs)
    # Directed-edge attribution: keep sign to know direction of effect
    dir_imp = attr.sum(dim=1) if signed else attr.abs().sum(dim=1)  # [E_dir]

    # Average i→j and j→i into undirected bonds
    src, dst = data.edge_index
    idx_map = {(int(src[k]), int(dst[k])): k for k in range(src.size(0))}
    seen = set()
    bonds = []
    for k in range(src.size(0)):
        if k in seen:
            continue
        i, j = int(src[k]), int(dst[k])
        if (j, i) in idx_map:
            kr = idx_map[(j, i)]
            imp = 0.5 * (dir_imp[k] + dir_imp[kr])
            seen.add(kr)
        else:
            imp = dir_imp[k]
        seen.add(k)
        bonds.append((i, j, float(imp.item())))
    return bonds
