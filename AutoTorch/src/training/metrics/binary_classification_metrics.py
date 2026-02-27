"""
Binary classification metrics accumulator.
Tracks TP/TN/FP/FN and computes accuracy, APCER, BPCER.
"""
import torch
from typing import Dict, List
from .base_metrics import BaseMetrics


class BinaryClassificationMetrics(BaseMetrics):
    """
    Metrics for binary classification.

    Example:
        metrics = BinaryClassificationMetrics(threshold=0.5)
        metrics.reset_and_initialize()
        for X, y in dataloader:
            y_pred = model(X)
            metrics.update_counts_from_preds(y_pred, y)
        results = metrics.compute_metrics() # {"acc": 0.95, "apcer": 0.02, "bpcer": 0.03}
       
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset_and_initialize()

    def reset_and_initialize(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update_counts_from_preds(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Update counts with batch predictions.

        Args:
            y_pred: Model output (logits). Shape: [batch, 1]
            y_true: Ground truth labels (0 or 1). Shape: [batch]
        """
        y_pred = y_pred.squeeze(-1) # squeeze the last dimension
        probs = torch.sigmoid(y_pred)
        preds = (probs > self.threshold).int()
        y_true = y_true.int()

        self.tp += ((preds == 1) & (y_true == 1)).sum().item()
        self.tn += ((preds == 0) & (y_true == 0)).sum().item()
        self.fp += ((preds == 1) & (y_true == 0)).sum().item()
        self.fn += ((preds == 0) & (y_true == 1)).sum().item()

    def compute_metrics(self) -> Dict[str, float]:
        """Compute accuracy, APCER, BPCER from accumulated counts."""
        total = self.tp + self.tn + self.fp + self.fn
        acc = (self.tp + self.tn) / total if total > 0 else 0.0

        # APCER = FP / (TN + FP) - Attack Presentation Classification Error Rate
        # How often we incorrectly accept an attack (spoof) as genuine
        apcer = self.fp / (self.tn + self.fp) if (self.tn + self.fp) > 0 else -1.0

        # BPCER = FN / (TP + FN) - Bona Fide Presentation Classification Error Rate
        # How often we incorrectly reject a genuine as attack
        bpcer = self.fn / (self.tp + self.fn) if (self.tp + self.fn) > 0 else -1.0

        return {"acc": acc, "apcer": apcer, "bpcer": bpcer}

    def get_sync_values(self) -> List[float]:
        """Values to sync across DDP ranks."""
        return [float(self.tp), float(self.tn), float(self.fp), float(self.fn)]

    def load_sync_values(self, values: List[float]) -> None:
        """Load synced values after all_reduce."""
        self.tp, self.tn, self.fp, self.fn = [int(v) for v in values]
