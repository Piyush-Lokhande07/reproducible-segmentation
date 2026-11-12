import os
import csv
import json
import time
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import set_seed
from src.dataset import build_dataloaders
from src.metrics import Evaluator
from models.baseline import BaselineUNet
from models.modified import AttentionUNet

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Standard 21-class PASCAL VOC RGB colormap
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)


def decode_segmap(mask: np.ndarray) -> np.ndarray:
    """Converts a 2D class-index mask into an RGB image (ignore index 255 -> white)."""
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(VOC_COLORMAP))
    rgb[valid] = VOC_COLORMAP[mask[valid]]
    rgb[mask == 255] = [255, 255, 255]
    return rgb


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverts ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std) + mean
    return np.clip(img, 0.0, 1.0)


@torch.no_grad()
def benchmark_latency(model: torch.nn.Module, device: torch.device, batch_size: int = 1, img_size: int = 256, warmup: int = 50, runs: int = 200) -> dict:
    """Measures hardware-synchronized inference latency (ms) and throughput (FPS)."""
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    for _ in range(warmup):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings_ms = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)

    mean_ms = float(np.mean(timings_ms))
    std_ms = float(np.std(timings_ms))
    fps = float((batch_size * 1000.0) / mean_ms)

    return {
        "batch_size": batch_size,
        "resolution": f"{img_size}x{img_size}",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "mean_latency_ms": round(mean_ms, 3),
        "std_latency_ms": round(std_ms, 3),
        "throughput_fps": round(fps, 2)
    }


def compute_single_image_miou(pred: np.ndarray, target: np.ndarray, num_classes: int = 21, ignore_index: int = 255) -> float:
    """Computes foreground-aware mIoU for a single image to rank success/failure cases."""
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (pred == cls) & (target != ignore_index)
        target_inds = (target == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def save_comparison_grid(samples: list, save_path: str, title: str) -> None:
    """Saves a 4-column visual comparison grid: Source | Ground Truth | Baseline | Modified."""
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Source Image", "Ground Truth (White=Void)", "Baseline U-Net", "Modified (Attention Skip)"]
    for row_idx, sample in enumerate(samples):
        imgs = [
            sample["image"],
            decode_segmap(sample["gt"]),
            decode_segmap(sample["base_pred"]),
            decode_segmap(sample["mod_pred"])
        ]
        for col_idx, img in enumerate(imgs):
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=11, fontweight="bold")
        axes[row_idx, 0].text(
            5, 18, f"Base IoU: {sample['base_miou']:.2f} | Mod IoU: {sample['mod_miou']:.2f}",
            color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.7, pad=2)
        )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training_curves(base_csv: str, mod_csv: str, save_path: str) -> None:
    """Plots validation loss and mIoU curves across epochs for both runs."""
    def read_log(path):
        epochs, t_loss, v_loss, miou = [], [], [], []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                t_loss.append(float(row["train_loss"]))
                v_loss.append(float(row["val_loss"]))
                miou.append(float(row["mIoU"]))
        return epochs, t_loss, v_loss, miou

    if not (os.path.exists(base_csv) and os.path.exists(mod_csv)):
        return

    b_ep, b_tl, b_vl, b_miou = read_log(base_csv)
    m_ep, m_tl, m_vl, m_miou = read_log(mod_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(b_ep, b_vl, label="Baseline Val Loss", linestyle="--", color="#1f77b4")
    ax1.plot(m_ep, m_vl, label="Modified Val Loss", linewidth=2, color="#d62728")
    ax1.set_title("Validation Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(b_ep, b_miou, label="Baseline Val mIoU", linestyle="--", color="#1f77b4")
    ax2.plot(m_ep, m_miou, label="Modified Val mIoU", linewidth=2, color="#2ca02c")
    ax2.set_title("Validation Mean IoU (mIoU)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    os.makedirs("logs/evidence", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = build_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["evaluation"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        download=False
    )

    # Load both trained models
    baseline = BaselineUNet(num_classes=cfg["data"]["num_classes"], pretrained=False).to(device)
    modified = AttentionUNet(num_classes=cfg["data"]["num_classes"], pretrained=False).to(device)

    baseline.load_state_dict(torch.load("checkpoints/baseline_best.pth", map_location=device))
    modified.load_state_dict(torch.load("checkpoints/modified_best.pth", map_location=device))
    baseline.eval()
    modified.eval()

    base_eval = Evaluator(num_classes=cfg["data"]["num_classes"], ignore_index=cfg["data"]["ignore_index"])
    mod_eval = Evaluator(num_classes=cfg["data"]["num_classes"], ignore_index=cfg["data"]["ignore_index"])

    sample_records = []

    for images, masks in tqdm(val_loader, desc="Evaluating Validation Set"):
        images_dev = images.to(device)
        base_preds = torch.argmax(baseline(images_dev), dim=1).cpu()
        mod_preds = torch.argmax(modified(images_dev), dim=1).cpu()

        base_eval.update(base_preds, masks)
        mod_eval.update(mod_preds, masks)

        for i in range(images.size(0)):
            gt_np = masks[i].numpy()
            bp_np = base_preds[i].numpy()
            mp_np = mod_preds[i].numpy()
            b_iou = compute_single_image_miou(bp_np, gt_np)
            m_iou = compute_single_image_miou(mp_np, gt_np)
            sample_records.append({
                "image": unnormalize_image(images[i]),
                "gt": gt_np,
                "base_pred": bp_np,
                "mod_pred": mp_np,
                "base_miou": b_iou,
                "mod_miou": m_iou,
                "delta": m_iou - b_iou
            })

    base_res = base_eval.compute()
    mod_res = mod_eval.compute()

    # Save per-class IoU table
    with open("logs/evidence/per_class_iou.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "baseline_iou", "modified_iou", "delta_iou"])
        for idx, name in enumerate(VOC_CLASSES):
            b_val = base_res["per_class_iou"][idx]
            m_val = mod_res["per_class_iou"][idx]
            writer.writerow([idx, name, round(b_val, 4), round(m_val, 4), round(m_val - b_val, 4)])

    # Parameter counts & Latency benchmarks
    base_params = sum(p.numel() for p in baseline.parameters())
    mod_params = sum(p.numel() for p in modified.parameters())

    summary = {
        "baseline": {
            "mIoU": round(base_res["mIoU"], 4),
            "parameters": base_params,
            "latency_bs1": benchmark_latency(baseline, device, batch_size=1, img_size=cfg["data"]["img_size"]),
            "latency_bs16": benchmark_latency(baseline, device, batch_size=16, img_size=cfg["data"]["img_size"]),
        },
        "modified_attention_skip": {
            "mIoU": round(mod_res["mIoU"], 4),
            "parameters": mod_params,
            "param_overhead_pct": round(((mod_params - base_params) / base_params) * 100, 2),
            "latency_bs1": benchmark_latency(modified, device, batch_size=1, img_size=cfg["data"]["img_size"]),
            "latency_bs16": benchmark_latency(modified, device, batch_size=16, img_size=cfg["data"]["img_size"]),
        }
    }

    with open("logs/evidence/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=== FINAL EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # 1. Largest positive improvements (Attention Skip helping boundary/clutter suppression)
    by_gain = sorted(sample_records, key=lambda x: x["delta"], reverse=True)
    save_comparison_grid(by_gain[:4], "logs/evidence/visual_improvements.png", "Qualitative Improvements (Largest Attention Gain)")

    # 2. Failure cases (Lowest modified mIoU across validation set)
    by_worst = sorted(sample_records, key=lambda x: x["mod_miou"])
    save_comparison_grid(by_worst[:4], "logs/evidence/failure_cases.png", "Qualitative Failure Cases (Lowest Validation IoU)")

    # 3. Training & Validation curves
    plot_training_curves("logs/baseline_training_log.csv", "logs/modified_training_log.csv", "logs/evidence/training_curves.png")
    print("\nAll evidence artifacts saved to logs/evidence/")


if __name__ == "__main__":
    main()
