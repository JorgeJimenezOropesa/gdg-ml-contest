# GDG UAM Machine Learning Contest — Real-Time Multivariate Time-Series Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU_Optimized-005CED?logo=onnx&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-NAS_%26_HPO-3776AB)

High-throughput, low-latency 1D Convolutional Residual Network deployed via single-threaded **ONNX Runtime** for multivariate sensor time-series classification under strict sequential CPU execution budgets.

---

## Overview & Problem Formulation

The **GDG UAM Machine Learning Contest** challenges teams to classify 4-channel sensor time series (`shape: (4, 256)`) across **8 target classes**, handling missing sensor readings (`NaN` gaps), heterogeneous sampling rates, and class imbalance.

Unlike standard offline benchmarks, submissions are evaluated in a **strictly sequential streaming harness** (`predict(sample)` called one sample at a time) on a single CPU thread within a fixed **60-second cumulative inference budget**. The competition ranking metric rewards both per-class precision and inference throughput via a class-wise geometric mean:

$$\text{Score} = \left( \prod_{k=0}^{K-1} \max\!\left(\epsilon,\; n_k^{\alpha} \cdot \frac{c_k - \lambda w_k}{n_k} \cdot \left(\frac{c_k}{n_k}\right)^{p}\right) \right)^{\frac{1}{K}}$$

where $n_k$, $c_k$, and $w_k$ denote the total, correct, and wrong predictions for class $k$ within the time budget ($\alpha = 0.5$, $\lambda = 0.33$, $p = 2$, $\epsilon = 0.01$). Because higher inference speed directly increases $n_k$, architectural efficiency and runtime optimization are as critical as classification accuracy.

---

## Key Engineering Highlights

* **Time-Scale Alignment & Dual-Crop Test-Time Ensembling (`preprocess.py`)**: Input sequences arrive at different sampling rates (`sample_rate` $\in \{1, 2\}$). High-frequency series (`rate = 2`) are 2× temporally average-pooled to `(4, 128)`, while standard-rate series (`rate = 1`) use random 128-step crops during training and are split into two non-overlapping `(4, 128)` halves at inference—averaging logits across both windows in a single batched forward pass.
* **Custom Lightweight 1D ResNet (`net.py`)**: A pre-activation 1D Residual Network (`Stem` $\rightarrow$ 4 residual stages with strided downsampling $\rightarrow$ `Global Average Pooling` $\rightarrow$ `Linear` head) engineered to maximize receptive field while keeping single-thread CPU FLOPs minimal.
* **Single-Threaded ONNX Runtime Inference (`model.py`)**: Trained PyTorch weights are exported to **ONNX** and executed with `ORT_SEQUENTIAL`, full graph optimizations (`ORT_ENABLE_ALL`), and single-thread pinning (`intra_op_num_threads = 1`) to eliminate Python/PyTorch dispatch overhead.
* **Domain-Specific Data Augmentation & Balanced Sampling (`augmentations.py`, `dataset.py`)**: On-the-fly training augmentations simulate real-world sensor degradation—smooth polynomial drift, relative Gaussian noise, and stochastic `NaN` interval dropout—paired with a stratified `BatchSampler` ensuring uniform class representation per epoch.
* **Metric-Driven Architecture & Hyperparameter Search (`optuna_study.py`)**: Automated search using multivariate **TPE** sampling and **Hyperband** pruning, optimizing network depth, channel widths, kernel sizes, and regularization directly against a local replica of the official time-budgeted harness (`exam.py`).

---

## Repository Structure

```text
├── model.py              # Production inference entrypoint (ONNX Runtime CPU session)
├── net.py                # Custom pre-activation 1D ResNet architecture (ConvNet)
├── preprocess.py         # Linear NaN interpolation, sample-rate alignment & Z-score norm
├── augmentations.py      # On-the-fly time-series augmentations (drift, noise, NaN masking)
├── config.py             # Immutable dataclass configuration for data augmentation
├── dataset.py            # Process-safe PyTorch Dataset & class-balanced BatchSampler
├── train.py              # Training pipeline (AdamW, OneCycleLR) & ONNX graph export
├── exam.py               # Local implementation of the sequential time-budgeted evaluator
├── split.py              # Stratified 90k / 10k train-validation split generator
├── optuna_study.py       # Automated Neural Architecture & Hyperparameter Search
├── optuna_config.yaml    # Search space, TPE sampler & Hyperband pruner specification
└── requirements.txt      # Project dependencies

```

---

## Quickstart

### 1. Setup & Data Preparation

Install dependencies and generate the stratified 90k/10k train/validation split from the competition dataset:

```bash
pip install -r requirements.txt
python split.py

```

### 2. Training & ONNX Export

Train the 1D ResNet with class-weighted Cross-Entropy, evaluate against the local time-budgeted benchmark, and export the best checkpoint to `model.onnx`:

```bash
python train.py

```

### 3. Hyperparameter & Architecture Search (Optional)

Run the Optuna study to explore architectures and training hyperparameters:

```bash
python optuna_study.py --config optuna_config.yaml

```

### 4. Official Harness Evaluation

Benchmark the exported ONNX model under the exact sequential constraints of the competition:

```bash
python harness.py evaluate --model model.py --source file:data/val_split.npz

```

---

## Team

* **Inhar Castro**
* **Miguel Campo**
* **Jorge Jiménez**
