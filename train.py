import os
import csv
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils import set_seed, log_environment_info
from src.dataset import build_dataloaders
from src.metrics import Evaluator
from models.baseline import BaselineUNet
from models.modified import AttentionUNet

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, evaluator, device):
    model.eval()
    evaluator.reset()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        evaluator.update(preds, masks)
        
    metrics = evaluator.compute()
    metrics["val_loss"] = total_loss / len(loader)
    return metrics

def run_experiment(config_path: str = "configs/config.yaml", model_variant: str = "baseline"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Enforce strict reproducibility
    set_seed(cfg["training"]["seed"])
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_environment_info(f"logs/{model_variant}_env_versions.txt")

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Executing [{model_variant}] on device: {device}")

    # 2. Build DataLoaders
    train_loader, val_loader = build_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        download=False
    )

    # 3. Model setup
    if model_variant == "baseline":
        model = BaselineUNet(num_classes=cfg["data"]["num_classes"]).to(device)
    elif model_variant == "modified":
        model = AttentionUNet(num_classes=cfg["data"]["num_classes"]).to(device)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    # 4. Optimizer, Scheduler, Stopping Rule
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_index"])
    optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # 5. Training log file
    log_file = f"logs/{model_variant}_training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "mIoU", "lr"])

    evaluator = Evaluator(num_classes=cfg["data"]["num_classes"], ignore_index=cfg["data"]["ignore_index"])
    best_miou = 0.0
    patience = 8
    patience_counter = 0

    # 6. Training Loop
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, evaluator, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_miou = val_metrics["mIoU"]
        val_loss = val_metrics["val_loss"]

        print(f"Epoch {epoch:02d}/{cfg['training']['epochs']:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_miou, current_lr])

        # Stopping rule / checkpoint tracking
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            torch.save(model.state_dict(), f"checkpoints/{model_variant}_best.pth")
            print(f"==> Checkpoint saved: New best mIoU {best_miou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "modified"])
    args = parser.parse_args()
    run_experiment(model_variant=args.model)
