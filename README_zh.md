# PCF 主方法与两个复现数据集

本仓库对应论文 Beyond the First Choice: Unsupervised Fusion of Vision–Language Model Predictions。

## 安装和运行

使用 Python 3.12，CPU 即可运行。在仓库根目录执行以下命令。

```text
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python reproduce.py --data-dir data --protocol both --output-dir results/reproduction
```

默认运行两个数据集、两种协议、2026–2030 五个种子，共 20 次评价。
快速检查可在复现命令中加入 `--seeds 2026`，运行四个组合。

## 包内数据

| 文件 | 论文任务 | 样本数 | VLM 数 | 类别数 |
|---|---|---:|---:|---:|
| data/cifar19.mat | CIFAR-19 | 10000 | 5 | 19 |
| data/flower17.mat | Flower-15 | 1360 | 5 | 15 |

两个 MAT 文件包含冻结 VLM 的预测概率、模型顺序、类别名称和评价标签，可直接作为主方法输入。
`flower17.mat` 保留历史文件名，当前任务实际为 15 类。详细字段见 `data/README.md`。

## 两种协议

- Inductive 归纳式协议按种子打乱样本，80% 无标签样本用于拟合，20% 留出用于预测和评价。
- Transductive 传导式协议在完整无标签样本池上拟合和预测。
- 拟合和预测接口只接收概率。真实标签在预测文件保存后才读入，用于计算 ACC、Macro-F1 和 Brier。

单独运行某一协议时，将 `--protocol both` 改为 `--protocol inductive` 或 `--protocol transductive`。
单任务入口也保留，可以直接运行以下命令。

```text
python run.py --input data/flower17.mat --protocol inductive --seed 2026 --output results/flower_ind.npz --evaluate
python run.py --input data/flower17.mat --protocol transductive --seed 2026 --output results/flower_trans.npz --evaluate
python -m unittest discover -s tests -v
```

结果目录保存每次预测、样本和拟合索引、`metrics.csv` 与 `summary.json`。
汇总值为这两个任务的平均值，论文六任务平均值需使用全部六个任务计算。

核心算法及参数沿用正式实验冻结实现。已执行的复现核对与预期数值见 `REPRODUCIBILITY.md`。
