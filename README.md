# Molecule Interpretability (Tox21)
Grounded, assay-aware explanations for D-MPNN toxicity models — with Integrated Gradients (IG) → SMARTS concepts, a YAML knowledge base, threshold calibration, and faithfulness metrics.

> This repo trains a D-MPNN on **Tox21**, evaluates single/ensemble models, generates **KB-grounded** rationales, calibrates **operating thresholds** (F1-optimal or precision-targeted), and computes **faithfulness** scores (comprehensiveness, sufficiency, deletion/insertion AUC).

---

## TL;DR quickstart (macOS example)

```bash
# 0) Install Conda (Miniforge)
brew install --cask miniforge

# 1) Create & activate env (Python 3.11)
conda create -n molsxai python=3.11 -y
conda activate molsxai

# 2) Core deps (channels matter)
conda install -c conda-forge rdkit -y
conda install -c pytorch pytorch torchvision -y
conda install -c pyg pyg -y

# 3) Install this project and extras
pip install -r requirements.txt
pip install -e .

# 4) Sanity check: RDKit, Torch, PyG, and the package import
python - <<'PY'
from rdkit import Chem
import torch, torch_geometric, llm_mol_interp
print("RDKit OK:", bool(Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")))
print("Torch:", torch.__version__, "| PyG:", torch_geometric.__version__)
PY
```

---

## Data

Place your Tox21 CSV at `data/tox21_multitask.csv` and ensure:
- a SMILES column (default: `smiles`)
- one binary column per assay, e.g.  
  `NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PR, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP`

> If your file/column names differ, adjust `--data-csv`, `--smiles-col`, and the `--target-names` arguments where applicable.

---

## Train a D-MPNN (single seed)

```bash
python -m llm_mol_interp.cli.train_dmpnn   --data-csv data/tox21_multitask.csv --smiles-col smiles   --out-dir runs/dmpnn_tox21   --epochs 50 --batch-size 64 --depth 3 --hidden 256   --dropout 0.2 --lr 1e-3 --scheduler onecycle   --weight-decay 1e-2 --clip-grad 1.0 --patience 10 --seed 7
```

### Reproduce 5 seeds for an ensemble

```bash
for SEED in 1 2 3 4 5; do
  python -m llm_mol_interp.cli.train_dmpnn     --data-csv data/tox21_multitask.csv --smiles-col smiles     --out-dir runs/dmpnn_tox21_seed${SEED}     --epochs 50 --batch-size 64 --depth 3 --hidden 256     --dropout 0.2 --lr 1e-3 --scheduler onecycle     --weight-decay 1e-2 --clip-grad 1.0 --patience 10 --seed ${SEED}
done
```

---

## Ensemble evaluation

```bash
python -m llm_mol_interp.cli.ensemble_eval   --data-csv data/tox21_multitask.csv --smiles-col smiles   --ckpts runs/dmpnn_tox21_seed1/best.pt          runs/dmpnn_tox21_seed2/best.pt          runs/dmpnn_tox21_seed3/best.pt          runs/dmpnn_tox21_seed4/best.pt          runs/dmpnn_tox21_seed5/best.pt   --split-seed 7 --batch-size 256   --out-dir runs/dmpnn_tox21_ensemble --save-preds-csv
```

---

## Generate grounded explanations

The explainer uses **Integrated Gradients** from a reference, aggregates atom/edge attributions into fragments, matches to **SMARTS** patterns, and links to a YAML KB (`llm_mol_interp/xai/kb.yaml`) to produce **assay-aware rationales**.

```bash
python -m llm_mol_interp.cli.explain_testset   --data-csv data/tox21_multitask.csv --smiles-col smiles   --ckpt runs/dmpnn_tox21_seed3/best.pt   --ensemble-ckpts runs/dmpnn_tox21_seed1/best.pt                    runs/dmpnn_tox21_seed2/best.pt                    runs/dmpnn_tox21_seed3/best.pt                    runs/dmpnn_tox21_seed4/best.pt                    runs/dmpnn_tox21_seed5/best.pt   --split-seed 7 --max-mols -1 --ig-steps 64   --kb llm_mol_interp/xai/kb.yaml   --out-dir runs/dmpnn_tox21_ensemble/explanations
```

Outputs (in `.../explanations/`):
- `explanations.jsonl` with per-molecule predictions, attribution maps, matched SMARTS, and generated rationales
- auxiliary index maps for traceability

---

## Calibrate thresholds & report verdicts

Calibrate **operating points** (e.g., target precision for screening) and/or compute **F1-optimal** thresholds.

```bash
python -m llm_mol_interp.utils.calibrate_threshold   --jsonl runs/dmpnn_tox21_ensemble/explanations/explanations.jsonl   --csv data/tox21_multitask.csv   --smiles-col smiles   --target-precision 0.50   --out-dir runs/dmpnn_tox21_ensemble
```

Generate a short report at **F1-optimal** thresholds:

```bash
python -m llm_mol_interp.utils.verdict_report   --explanations-jsonl runs/dmpnn_tox21_ensemble/explanations/explanations.jsonl   --data-csv data/tox21_multitask.csv --smiles-col smiles   --thresholds-json runs/dmpnn_tox21_ensemble/thresholds_f1.json
```

Score with a chosen threshold set:

```bash
python -m llm_mol_interp.utils.score_with_threshold   --jsonl runs/dmpnn_tox21_ensemble/explanations/explanations.jsonl   --csv data/tox21_multitask.csv   --smiles-col smiles   --thresholds runs/dmpnn_tox21_ensemble/thresholds_f1.json
```

> **Note:** If you used the calibration step, threshold JSONs (e.g., `thresholds_f1.json`, precision-targeted) will be saved under the same `--out-dir`.

---

## Single-molecule / small-batch mode

Create `/tmp/one.csv` with a column `smiles`, then:

```bash
python -m llm_mol_interp.cli.explain_testset   --data-csv /tmp/one.csv --smiles-col smiles   --ckpt runs/dmpnn_tox21_seed3/best.pt   --ig-steps 64   --kb llm_mol_interp/xai/kb.yaml   --out-dir runs/dmpnn_tox21_explanations_single   --thresholds-json runs/dmpnn_tox21_ensemble/thresholds_f1.json   --default-thr 0.5
```

---

## Faithfulness (masking & metrics)

1) Re-score importances by **feature masking** (top-k, zeroing):

```bash
python llm_mol_interp/runtime/make_importances_from_masking.py   --in-jsonl runs/dmpnn_tox21_ensemble/explanations/explanations.jsonl   --out-jsonl runs/dmpnn_tox21_ensemble/explanations/explanations_masking.jsonl   --endpoint NR-AR   --ckpts runs/dmpnn_tox21_seed1/best.pt runs/dmpnn_tox21_seed2/best.pt           runs/dmpnn_tox21_seed3/best.pt runs/dmpnn_tox21_seed4/best.pt           runs/dmpnn_tox21_seed5/best.pt   --target-names NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PR NR-PPAR-gamma SR-ARE SR-ATAD5 SR-HSE SR-MMP   --topk 10 --mask-mode zero
```

2) Compute **comprehensiveness**, **sufficiency**, **deletion/insertion AUC**, etc.:

```bash
python -m chemxai_faithfulness.cli   --explanations-jsonl runs/dmpnn_tox21_ensemble/explanations/explanations_masking.jsonl   --ckpts runs/dmpnn_tox21_seed1/best.pt runs/dmpnn_tox21_seed2/best.pt           runs/dmpnn_tox21_seed3/best.pt runs/dmpnn_tox21_seed4/best.pt           runs/dmpnn_tox21_seed5/best.pt   --endpoint NR-AR   --out-dir runs/dmpnn_tox21_ensemble/faithfulness_masking   --kb llm_mol_interp/xai/kb.yaml   --target-names NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PR NR-PPAR-gamma SR-ARE SR-ATAD5 SR-HSE SR-MMP
```

---

## Example results

**Per-task metrics (F1-optimal thresholds):**
```
            task    n  tp   fp   tn  fn  precision  recall     f1  accuracy  threshold
10         SR-MMP  517  69   78  320  50      0.469   0.580  0.519     0.752      0.590
7          SR-ARE  499  66  152  245  36      0.303   0.647  0.413     0.623      0.630
5       NR-ER-LBD  615  19   32  534  30      0.373   0.388  0.380     0.899      0.515
9          SR-HSE  565  18   39  487  21      0.316   0.462  0.375     0.894      0.705
4           NR-ER  537  39   88  362  48      0.307   0.448  0.364     0.747      0.540
1       NR-AR-LBD  593  15   44  522  12      0.254   0.556  0.349     0.906      0.585
2          NR-AhR  574  45  118  356  55      0.276   0.450  0.342     0.699      0.410
0           NR-AR  652  16   49  574  13      0.246   0.552  0.340     0.905      0.665
3    NR-Aromatase  488  15   51  407  15      0.227   0.500  0.312     0.865      0.755
11         SR-p53  590  34  149  383  24      0.186   0.586  0.282     0.707      0.615
6   NR-PPAR-gamma  539   8   71  446  14      0.101   0.364  0.158     0.842      0.550
8        SR-ATAD5  625   3    5  587  30      0.375   0.091  0.146     0.944      0.695

Macro F1: 0.332 | Micro F1: 0.362 | Micro Acc: 0.820
```

**Faithfulness summary (IG-based):**
```json
{
  "N": 708,
  "comprehensiveness_mean": 0.14928048246908518,
  "sufficiency_mean": 0.3203860940911549,
  "deletion_auc_mean": 0.20598602086696105,
  "insertion_auc_mean": 0.31909215035897726
}
```

---

## Extra: explain one model then score with thresholds

```bash
# A) Run explanations for ONE model (e.g., seed 3)
python -m llm_mol_interp.cli.explain_testset   --data-csv data/tox21_multitask.csv   --smiles-col smiles   --ckpt runs/dmpnn_tox21_seed3/best.pt   --out-dir runs/dmpnn_tox21_seed3/explanations

# Produce F1-threshold scores for those explanations
python -m llm_mol_interp.utils.score_with_threshold   --jsonl runs/dmpnn_tox21_seed3/explanations/explanations.jsonl   --csv data/tox21_multitask.csv   --smiles-col smiles   --thresholds runs/dmpnn_tox21_ensemble/thresholds_f1.json
```

---

## Tips & troubleshooting

- **Conda channels**: install `rdkit` from **conda-forge**, PyTorch from the **pytorch** channel, and PyG from **pyg**.
- **macOS (Apple Silicon)**: CPU and MPS backends work; CUDA is Linux/NVIDIA only.
- **Reproducibility**: set `--split-seed` and `--seed` (per run); we ensemble 5 seeds by default.
- **KB file**: customize `llm_mol_interp/xai/kb.yaml` to extend SMARTS → concept mappings and rationale templates.

---

## Repository highlights

- `llm_mol_interp/cli/train_dmpnn.py` — train D-MPNN
- `llm_mol_interp/cli/ensemble_eval.py` — evaluate an ensemble
- `llm_mol_interp/cli/explain_testset.py` — run IG + SMARTS + KB rationales
- `llm_mol_interp/utils/calibrate_threshold.py` — calibrate operating points
- `llm_mol_interp/utils/verdict_report.py` — generate thresholded report
- `llm_mol_interp/utils/score_with_threshold.py` — apply thresholds to explanations
- `llm_mol_interp/runtime/make_importances_from_masking.py` — faithfulness via masking
- `llm_mol_interp/xai/kb.yaml` — knowledge base for assay-aware, grounded explanations

---



