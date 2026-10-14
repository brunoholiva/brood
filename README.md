# brood

> This is an in-progress repository. It will change.

This repository provides the code for an investigation on predictor models for **antibiotic activity**, with the following steps:

1. Split a collected (usually from literature) antimicrobial activity dataset, using generic scaffolds to split a 10% set that **will not be used** until the end of analysis.

2. Perform a screening with model architectures and algorithms (all using ECFPs as a representation if needed).

3. The best performing algorithms are selected to be trained with a bunch of molecular representations.

4. Both the algorithm screening and the molecular representations are evaluated with 5x5-fold cross-validation on the 90% data. The held-out test set is used only at the end, after all decisions have been made.

---

## Polaris Benchmarking

Some interesting combinations are being applied to datasets beyond antibiotic activity. The [Polaris benchmark framework](https://polarishub.io/) is interesting for that: it's highly curated, easily testable, and since its tasks are varied, we can have a sense if a strategy works for other domains. For now, [TabPFN 3.5](https://github.com/PriorLabs/TabPFN) with either [MiniMol](https://github.com/graphcore-research/minimol) or [CheMeleon](https://github.com/JacksonBurns/chemeleon) learned representations is being employed.

---

## Screening

All screening models use Morgan ECFP4 with 2048 bits, except when the algorithm is dependent on a specific representation (such as Chemprop).

- Logistic regression
- Random forest
- LightGBM
- [TabPFN v3.5](https://github.com/PriorLabs/TabPFN)
- [Chemprop v2](https://github.com/chemprop/chemprop)
- Max-similarity baseline

Tunable models receive the same tuning budget: 10 Optuna trials per fold (which is internally split)
---

## Representations

| Representation | Dims | Notes |
|:---------------|-----:|:------|
| Morgan ECFP4 2048 | 2048 | screening representation|
| Morgan ECFP4 512 | 512 |  |
| MACCS | 167 | predefined structural keys |
| RDKit2D | 200 | 0/1D/2D physicochemical descriptors from descriptastorus |
| CheMeleon | 2048 | pretrained chemprop MPNN embeddings |
| Morgan + MACCS + RDKit2D | 879 | concatenated fingerprint and descriptor block |
| MiniMol | 512 | learned GNN representation |

---

## Environments

Create the conda environments with micromamba/conda, then install the package
(editable) in whichever ones run Python code — the `brood-*` entry points are
defined by the editable install:

```bash
micromamba create -f envs/brood.yml
micromamba activate brood
pip install -e .
```

| Env | Models / representations |
|-----|--------------------------|
| `brood` | base env: logistic regression, RF, LightGBM, max-similarity baseline; Morgan/MACCS/RDKit2D/combined features |
| `tabpfn` | TabPFN v3.5 (non-chemeleon representations) |
| `chemprop` | Chemprop v2 MPNN |
| `chemeleon` | CheMeleon fingerprints + TabPFN |
| `minimol` | MiniMol fingerprints + TabPFN |

Each env has its own file in `envs/` (`brood.yml`, `tabpfn.yml`, `chemprop.yml`,
`chemeleon.yml`, `minimol.yml`).

This is a temporary and messy solution.
---

## Datasets (antibiotics)

For now the antibiotic study uses the following datasets:

| Dataset | Source | Description |
|:--------|:-------|:------|
| `GNEtolC` | [Paper](https://www.nature.com/articles/s41587-025-02814-6)  | *E. coli* Delta-tolC strain

Raw CSVs live in `data/raw/`.


## How it works

| Command | What it does | Writes |
|---------|--------------|--------|
| `brood-split` | generic scaffold based train/test split + repeat and fold information `data/raw/` | `data/splits/<dataset>/train.csv`, `test.csv`, `cv_folds.csv` |
| `brood-run` | YAML-driven pipeline stage (screening, representation, held-out) | per-run artifacts + figures (see steps below) |
| `brood-polaris` | train TabPFN on a Polaris `tdcommons` benchmark, evaluate locally | `data/polaris_results/<task>/tabpfn_<rep>/` |
| `brood-train` | one model on one dataset (tune, `--cv`, or held-out predict) | `data/runs/<dataset>/<model>/` |
| `brood-evaluate` | score prediction files against a held-out test set | `data/evaluations/<dataset>/` |
| `brood-plot-cv` | CV comparison figures from `metrics/cv.csv` files | `figures/<dataset>/cv/*.svg` |

Outputs under `data/` and `figures/`:

```
data/
├── raw/                  input CSVs
├── splits/<dataset>/     train.csv · test.csv · cv_folds.csv
├── features/<dataset>/<representation>/   cached .npy
├── runs/<dataset>/<run>/ params.json · predictions/{cv,test}.csv · metrics/cv.csv
├── evaluations/<dataset>/ merged_results.csv · metrics.csv
└── polaris_results/<task>/tabpfn_<rep>/ predictions.csv · metrics.csv · metadata.json
figures/<dataset>/{cv,heldout}/
```
---

## Comparing models

Every run is scored on the same 5 repeats × 5 folds of cross-validation, so all runs see the same splits and can be compared run by run. The CV figures rank them by average precision (AP): `<dataset>_cv_ap.svg` ranks the screening runs, `<dataset>_cv_representation.svg` compares representations, and `<dataset>_cv_representation_ap.svg` ranks the representation runs.

Whether two runs really differ is judged with a repeated-measures test. By default (`--test auto`) this is a repeated-measures ANOVA with Tukey HSD; when its assumptions do not hold, it falls back to the rank-based Friedman test with Conover-Holm corrections. `--test parametric` and `--test nonparametric` force one path.

---

## Reproducing the study (GNEtolC)

1. Activate `brood` and install the package (see Environments).
2. Split the raw data:
   ```bash
   brood-split
   ```
   Do not re-run this later — it would invalidate the fixed CV folds.
3. Screen models on ECFP4/2048:
   ```bash
   brood-run --config configs/screening.yaml --dataset GNEtolC
   ```
   Inspect `figures/GNEtolC/cv/GNEtolC_cv_ap.svg` and confirm the best algorithms.
4. Compare representations for those models:
   ```bash
   brood-run --config configs/representation.yaml --dataset GNEtolC
   ```
   Rank `figures/GNEtolC/cv/GNEtolC_cv_representation_ap.svg` and set the top
   three runs in `configs/heldout.yaml`.
5. Score the selected runs once on the held-out test:
   ```bash
   brood-run --config configs/heldout.yaml --dataset GNEtolC
   ```
   Inspect `figures/GNEtolC/heldout/GNEtolC_heldout_pr.svg`.

---

## Polaris benchmarks

```bash
micromamba activate minimol   # or tabpfn/chemeleon, per representation
brood-polaris --config configs/polaris.yaml --dataset ames --representation rdkit2d
```

`configs/polaris.yaml` maps each dataset to its task and metric; splits and
metrics are owned by Polaris. The runner only saves local artifacts
(`metadata.json`, `metrics.csv`, unlabeled `predictions.csv`) so it never
uploads. Inspect and upload via `polaris_results.ipynb`: log in, dry-run one
cell per dataset. Uploads reuse the
`BenchmarkResults` in `metadata.json`, so no test target is ever read.

---

## Stage configs

`configs/screening.yaml`, `configs/representation.yaml`, and
`configs/heldout.yaml` share a schema:

- `defaults: {n_trials}` — Optuna tuning budget for trainable runs.
- `datasets` — datasets to run;
- `models` — one entry per run:
  ```yaml
  rf_morgan512:
    env: brood            # conda env that runs this model
    model: rf             # registry key (logreg/rf/lgbm/tabpfn/chemprop/similarity)
    representation: morgan512
    gpu: 0                # TabPFN/Chemprop GPUs
  ```
- `plot_models` / `representation_plot_models` — runs shown in the CV figures.
- `heldout_models` — the top CV runs refit on the full training split and
  scored once on `test.csv` (the only stage that reads it).

Stages skip outputs that already exist; a failed model logs an error and does
not abort the rest. `brood-run --force` reruns existing outputs.
