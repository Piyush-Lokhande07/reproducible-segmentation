import numpy as np
import torch

class Evaluator:
    """Computes exact per-class IoU and mIoU over a confusion matrix."""
    def __init__(self, num_classes: int = 21, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray) -> np.ndarray:
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_true != self.ignore_index)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        for lt, lp in zip(targets, preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def compute(self) -> dict:
        hist = self.confusion_matrix
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        
        denominator = tp + fp + fn
        iou_per_class = np.full(self.num_classes, np.nan)
        valid = denominator > 0
        iou_per_class[valid] = tp[valid] / denominator[valid]

        # Filter out NaN classes that were not present in the split
        present_iou = iou_per_class[~np.isnan(iou_per_class)]
        miou = float(np.mean(present_iou)) if len(present_iou) > 0 else 0.0

        return {
            "mIoU": miou,
            "per_class_iou": iou_per_class.tolist()
        }

    def reset(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
