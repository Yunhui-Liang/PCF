# Progressive Candidate Fusion (PCF)

Code for **Beyond the First Choice: Unsupervised Fusion of Vision–Language
Model Predictions**.

PCF combines first-choice decisions, candidate rankings and class probabilities
from frozen vision–language models through an unlabeled consensus and learned
source-specific prediction mappings. Ground-truth labels are used for evaluation.

## Installation

Use Python 3.12 and a virtual environment.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The method runs on CPU. A CPU-only PyTorch wheel can be installed before the
remaining requirements using the official CPU index.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

## Included datasets

The `data/` directory contains the processed frozen-VLM probability inputs and
evaluation labels for two paper tasks. They are ready to run.

| File | Task | Samples | Sources | Classes |
|---|---|---:|---:|---:|
| data/cifar19.mat | CIFAR-19 | 10000 | 5 | 19 |
| data/flower17.mat | Flower-15 | 1360 | 5 | 15 |

`flower17.mat` retains its experiment filename. Its task has 15 classes.
See [data/README.md](data/README.md) for variable definitions and source order.

## Reproduce both protocols

From the repository root, run the following command.

```bash
python reproduce.py --data-dir data --protocol both --output-dir results/reproduction
```

The defaults run both included datasets with seeds **2026–2030**, producing
20 evaluations with the paper's fixed main-method settings.

- **Inductive** uses an unlabeled 80% fitting split and a held-out 20% prediction
  split. The split uses `default_rng(seed).permutation(N)`, with the first
  `round(0.2*N)` indices held out and both index sets sorted.
- **Transductive** fits and predicts on the same complete unlabeled pool.
- **Metrics** include ACC and macro-F1 in percent, and Brier as the sample mean
  of summed squared errors over all classes. Dataset means weight the two tasks
  equally. Across-seed standard deviations use `ddof=1`.

Outputs include per-run probabilities, predictions, sample/fitting indices,
`metrics.csv` and `summary.json`. These summary values cover the two included
tasks. The paper's six-task averages use all six tasks.

A shorter check uses one seed for all four dataset/protocol combinations.

```bash
python reproduce.py --data-dir data --protocol both --seeds 2026 --output-dir results/quick
```

Individual protocols and datasets can also be selected.

```bash
python reproduce.py --data-dir data --protocol inductive --output-dir results/inductive
python reproduce.py --data-dir data --protocol transductive --output-dir results/transductive
python run.py --input data/flower17.mat --protocol inductive --seed 2026 --output results/flower_ind.npz --evaluate
python run.py --input data/flower17.mat --protocol transductive --seed 2026 --output results/flower_trans.npz --evaluate
python -m unittest discover -s tests -v
```

## Python API

```python
from pcf import PCF

# Aligned samples x sources x classes probabilities.
model = PCF(n_jobs=1, random_state=2026).fit(P_train)
q_test = model.predict_proba(P_test)
q_transductive = PCF().fit_predict_proba(P)
```

Labels are loaded for evaluation after fitting, prediction and prediction-file
creation. The fitting and prediction interfaces receive only source probabilities.

## Fixed main-method settings

| Component | Setting |
|---|---|
| Candidate rank limit | min(K, 12) |
| Rank / probability evidence weights | alpha = beta = 2 |
| Source correction | eta = 1.5 |
| Original/corrected posterior pooling | gamma = 0.45, temperature = 8 |
| Dirichlet consensus contribution | 0.1 |
| Per-source mappings | LightGBM, 45 estimators, 15 leaves, random_state = 123 |
| Mapping / consensus final mixing | lambda = 0.5 |

## Implementation and verification

- `pcf/model.py` provides the PCF fitting and prediction interface.
- `pcf/_core/` contains the frozen main-method computation with relative imports.
- `pcf/io.py` loads probability inputs, constructs splits and evaluates predictions.
- `run.py` and `reproduce.py` run one input or the two included tasks.
- `provenance/` records core-source fingerprints, dataset hashes and numerical checks.

The core retains its original V17Truth, V11Truth and V9Truth class names.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the executed verification and
expected results.
