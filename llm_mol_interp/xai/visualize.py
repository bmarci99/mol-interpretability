# --- add near top ---
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

def _norm_abs(vals, eps=1e-9):
    m = max((abs(v) for v in vals), default=0.0)
    return [0.0 if m < eps else abs(v)/m for v in vals], max(m, eps)

def draw_mol_with_highlight(
    smiles_or_mol,
    atom_ids=None,
    bond_pairs=None,     # list of (i, j, weight), weight may be signed
    out_path=None,
    size=(600, 450),
    signed=True,
):
    mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else Chem.MolFromSmiles(smiles_or_mol)
    rdDepictor.Compute2DCoords(mol)

    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opt = d.drawOptions()
    opt.addAtomIndices = False

    # build highlight lists
    hl_atoms = set(atom_ids or [])
    hl_bonds = []
    bond_colors, bond_radii = {}, {}

    if bond_pairs:
        amax = max((abs(w) for _,_,w in bond_pairs), default=1e-9)
        for i, j, w in bond_pairs:
            b = mol.GetBondBetweenAtoms(int(i), int(j))
            if b is None:
                continue
            bid = b.GetIdx()
            # color by sign, thickness by |w|
            color = (1.0, 0.0, 0.0) if (signed and w >= 0) else ((0.0, 0.0, 1.0) if signed else (1.0, 0.0, 0.0))
            radius = 0.10 + 0.25 * (abs(w) / amax)
            bond_colors[bid] = color
            bond_radii[bid]  = radius
            hl_bonds.append(bid)
            hl_atoms.update([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])

    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol,
        highlightAtoms=list(hl_atoms),
        highlightBonds=hl_bonds,
        highlightAtomColors={i: (1, 0.8, 0.8) for i in hl_atoms},
        highlightBondColors=bond_colors,
        #highlightBondRadii=bond_radii,
    )
    d.FinishDrawing()
    png = d.GetDrawingText()
    if out_path:
        with open(out_path, "wb") as fh:
            fh.write(png)
    return png
