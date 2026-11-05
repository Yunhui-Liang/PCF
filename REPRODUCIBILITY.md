# Reproducibility

This release was extracted into a separate directory and installed as an editable
Python package. The documented command below ran using only the two MAT inputs
in that extracted package. PYTHONPATH was removed and the user site was disabled.

```bash
python reproduce.py --data-dir data --protocol both --output-dir results/reproduction
```

## Verified evaluations

All 20 dataset/protocol/seed combinations completed. Predictions were compared
against archived full-method outputs from the corresponding original runs.
Predicted class IDs, prediction indices and inductive fitting indices matched
in every case. The probability tolerance was 1e-6.

The largest absolute probability difference was **1.19209289551e-07**.

| Protocol | Task | Seed | Fit / evaluation samples | ACC (%) | Maximum probability difference |
|---|---|---:|---:|---:|---:|
| inductive | CIFAR-19 | 2026 | 8000 / 2000 | 56.550000 | 0.000e+00 |
| inductive | CIFAR-19 | 2027 | 8000 / 2000 | 56.200000 | 0.000e+00 |
| inductive | CIFAR-19 | 2028 | 8000 / 2000 | 56.200000 | 0.000e+00 |
| inductive | CIFAR-19 | 2029 | 8000 / 2000 | 55.450000 | 0.000e+00 |
| inductive | CIFAR-19 | 2030 | 8000 / 2000 | 56.250000 | 0.000e+00 |
| inductive | Flower-15 | 2026 | 1088 / 272 | 91.544118 | 0.000e+00 |
| inductive | Flower-15 | 2027 | 1088 / 272 | 92.647059 | 0.000e+00 |
| inductive | Flower-15 | 2028 | 1088 / 272 | 93.382353 | 0.000e+00 |
| inductive | Flower-15 | 2029 | 1088 / 272 | 95.220588 | 0.000e+00 |
| inductive | Flower-15 | 2030 | 1088 / 272 | 94.117647 | 0.000e+00 |
| transductive | CIFAR-19 | 2026 | 10000 / 10000 | 55.560000 | 1.192e-07 |
| transductive | CIFAR-19 | 2027 | 10000 / 10000 | 55.560000 | 1.192e-07 |
| transductive | CIFAR-19 | 2028 | 10000 / 10000 | 55.560000 | 1.192e-07 |
| transductive | CIFAR-19 | 2029 | 10000 / 10000 | 55.560000 | 1.192e-07 |
| transductive | CIFAR-19 | 2030 | 10000 / 10000 | 55.560000 | 1.192e-07 |
| transductive | Flower-15 | 2026 | 1360 / 1360 | 93.014706 | 5.960e-08 |
| transductive | Flower-15 | 2027 | 1360 / 1360 | 93.014706 | 5.960e-08 |
| transductive | Flower-15 | 2028 | 1360 / 1360 | 93.014706 | 5.960e-08 |
| transductive | Flower-15 | 2029 | 1360 / 1360 | 93.014706 | 5.960e-08 |
| transductive | Flower-15 | 2030 | 1360 / 1360 | 93.014706 | 5.960e-08 |

## Five-seed task results

ACC values below are the per-task five-seed mean and sample standard deviation
in percentage points (`ddof=1`).

| Protocol | Task | Mean ACC (%) | Sample SD |
|---|---|---:|---:|
| inductive | CIFAR-19 | 56.130000 | 0.407124 |
| inductive | Flower-15 | 93.382353 | 1.399958 |
| transductive | CIFAR-19 | 55.560000 | 0.000000 |
| transductive | Flower-15 | 93.014706 | 0.000000 |

The reproduction summary covers these two included tasks. The paper's six-task
means require the complete six-task set.

ACC, macro-F1 and Brier were also checked against metrics calculated from the
archived predictions. Both protocol summaries and their across-seed standard
deviations were independently recomputed.

## Preserved implementation

The main-method computation files are byte-identical to the previously verified
release. Its 21 required function, class and constant definitions preserve the
executable ASTs of the frozen original implementation. Fingerprints are recorded
in `provenance/core_manifest.json`. The default dataset list now contains the two
included tasks. Input MAT files are byte-identical to their archived local inputs.

Three integration tests passed, covering held-out probability validity, inference
batch invariance, unchanged inputs, canonical disjoint splits, and invalid or
unfitted input handling.

## Tested environment

- Python 3.12.14, Windows CPU execution.
- numpy 2.3.5
- scipy 1.17.1
- scikit-learn 1.7.2
- lightgbm 4.6.0
- torch 2.7.1

Floating-point differences can occur across CPU platforms and dependency versions.
Machine-readable results are saved in `provenance/numerical_verification.json`.
Dataset dimensions, source/class names and hashes are in `provenance/data_manifest.json`.
