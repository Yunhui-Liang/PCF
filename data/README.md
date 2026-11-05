# Included probability inputs

The two MAT files contain aligned, frozen VLM outputs used by PCF.

| File | Task | N | M | K |
|---|---|---:|---:|---:|
| cifar19.mat | CIFAR-19 | 10000 | 5 | 19 |
| flower17.mat | Flower-15 | 1360 | 5 | 15 |

`flower17.mat` preserves the experiment filename and contains 15 classes.

Each file contains the following variables.

- `P_cell` is a 1 x M cell array. Each cell contains an N x K class-probability
  matrix for one source, with consistent sample and class order.
- `model_names` gives the source order.
- `class_names` gives the class order.
- `gt` contains zero-based integer labels used for post-prediction evaluation.

The source order is MiniCPM-V-4_5, Phi-3.5-vision-instruct,
Qwen2.5-VL-7B-Instruct, SmolVLM2-2.2B-Instruct, and llava-v1.6-mistral-7b-hf.

`pcf/io.py` reads the probability arrays, constructs the reproducible evaluation
split and reads evaluation labels after predictions have been saved. PCF uses
these frozen probability inputs directly. Dataset hashes, model names and class
names are recorded in `provenance/data_manifest.json`.

The single-input runner also accepts an N x M x K `.npy` array or an `.npz`
file with `P` or `probabilities`. Optional NPZ evaluation labels use `y` or `gt`.
MAT/NPZ labels spanning 1 through K are converted to zero-based IDs by the loader.
