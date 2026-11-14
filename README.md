# Reproducible Semantic Segmentation: Isolated Skip-Fusion Ablation

A strictly controlled, reproducible semantic segmentation benchmark on **PASCAL VOC 2012** evaluating the isolated impact of replacing standard feature concatenation with **Additive Attention Gates (AG)** in a ResNet-18 U-Net encoder–decoder.

## 1. Repository Structure

```text
├── configs/
│   └── config.yaml                  # Fixed seed, resolution, optimizer & schedule contract
├── models/
│   ├── baseline.py                  # Standard ResNet-18 U-Net (concatenation skip fusion)
│   └── modified.py                  # Controlled Variant (Attention-Gated skip fusion)
├── notebooks/
│   └── kaggle_runner.ipynb          # Executed Kaggle T4 GPU training & evaluation notebook
├── src/
│   ├── dataset.py                   # VOC 2012 data contract & Albumentations pipeline
│   ├── metrics.py                   # Confusion-matrix per-class IoU & mIoU (ignores 255)
│   └── utils.py                     # Global seed locking & environment metadata logger
├── logs/
│   ├── baseline_env_versions.txt    # Exact Python, PyTorch, CUDA & cuDNN versions
│   ├── baseline_training_log.csv    # Epoch-by-epoch baseline loss, mIoU & LR
│   ├── modified_env_versions.txt    # Exact runtime metadata for modified run
│   ├── modified_training_log.csv    # Epoch-by-epoch modified loss, mIoU & LR
│   └── evidence/
│       ├── evaluation_summary.json  # mIoU, parameter counts & synchronized latency
│       ├── per_class_iou.csv        # 21-class IoU breakdown (Baseline vs. Modified)
│       ├── training_curves.png      # Validation loss & mIoU progression
│       ├── visual_improvements.png  # Qualitative side-by-side comparison grids
│       └── failure_cases.png        # Lowest-IoU validation failure mode analysis
├── train.py                         # Deterministic training & checkpointing script
├── evaluate.py                      # Standalone evaluation, latency profiler & visualizer
└── requirements.txt                 # Pinned Python dependencies
```

## 2. Data & Reproducibility Contract

* **Dataset:** PASCAL VOC 2012 Semantic Segmentation (`1,464` train / `1,449` validation images)
* **Classes:** 20 foreground classes + 1 background (`0–20`); boundary/ambiguous void label `255` excluded from loss and IoU
* **Resolution & Preprocessing:** `256 × 256` (bilinear image interpolation, nearest-neighbor mask interpolation, ImageNet normalization)
* **Determinism:** Fixed seed `42` across `random`, `numpy`, `torch`, and `CUDA`, with `cudnn.deterministic = True`
* **Training Budget:** 40 epochs, batch size `16`, `AdamW` (`lr=3e-4`, `weight_decay=1e-4`), `CosineAnnealingLR`, early stopping patience `8`

## 3. Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train Baseline and Controlled Variant
python train.py --model baseline
python train.py --model modified

# 3. Run Exact Evaluation, Latency Benchmark & Visual Evidence Generation
python evaluate.py
```

## 4. Quantitative Summary (Tesla T4 GPU, 256 × 256)

| Metric | Baseline (Concat Skip) | Modified (Attention Skip) | Delta |
| :--- | :---: | :---: | :---: |
| **Validation mIoU** | `0.5126` | `0.5137` | `+0.0011` (`+0.21%`) |
| **Parameters** | `14,321,781` | `14,456,705` | `+134,924` (`+0.94%`) |
| **Latency (Batch = 1)** | `7.457 ± 0.051 ms` | `7.941 ± 0.068 ms` | `+0.484 ms` |
| **Throughput (Batch = 1)** | `134.11 FPS` | `125.93 FPS` | `-8.18 FPS` |
| **Latency (Batch = 16)** | `43.571 ± 0.305 ms` | `49.224 ± 0.335 ms` | `+5.653 ms` |
| **Throughput (Batch = 16)** | `367.22 FPS` | `325.04 FPS` | `-42.18 FPS` |

## 5. Scope & What This Experiment Does Not Establish

* **Multi-Seed Statistical Variance:** Results reflect a single deterministic seed (`42`); a `+0.0011` mIoU gain falls within typical run-to-run variance and does not establish statistical significance across multiple random seeds.
* **High-Resolution Scaling:** Both architectures were evaluated at `256 × 256` input resolution; spatial attention gains on fine object boundaries may behave differently at native `512 × 512` resolution.
* **Cross-Backbone / Cross-Dataset Generalization:** Findings are scoped to a ResNet-18 encoder trained on the standard `1,464`-image VOC 2012 split without SBD (`trainaug`) extensions or heavier backbones.
