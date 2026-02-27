"""
Base class for metrics accumulators.
Pluggable modules that track task-specific metrics during training/eval.
"""
from abc import ABC, abstractmethod
from typing import Dict, List
import torch


class BaseMetrics(ABC):
    """
    Abstract base class for task-specific metrics computation.

    This class defines the interface for metrics that accumulate counts across
    batches and compute final metrics at the end of an epoch. It supports
    distributed training (DDP) by providing sync methods for all_reduce.

    Lifecycle:
        1. reset_and_initialize() - Called at the start of each epoch to clear counters
        2. update_counts_from_preds() - Called after each batch to accumulate counts
        3. get_sync_values() / load_sync_values() - Used for DDP synchronization
        4. compute_metrics() - Called at the end of epoch to get final metrics

    Usage with BaseTrainer:
        from training.metrics import BinaryClassificationMetrics
        from training.trainers import BaseTrainer

        metrics = BinaryClassificationMetrics(threshold=0.5)
        trainer = BaseTrainer(model, loss_fn, optimizer, metrics, device)
        results = trainer.fit(train_loader, test_loader, epochs=10)

    How to Implement:
        1. Define internal counters (e.g., self.tp, self.tn, self.fp, self.fn)
        2. reset_and_initialize() should set all counters to 0
        3. update_counts_from_preds() should update counters from batch predictions
        4. compute_metrics() should compute final metrics from counters
        5. get_sync_values() should return counters as a list for DDP sync
        6. load_sync_values() should restore counters from synced list

    Example Implementation (Multi-class Accuracy):
        class MultiClassMetrics(BaseMetrics):
            def __init__(self):
                self.reset_and_initialize()

            def reset_and_initialize(self) -> None:
                self.correct = 0
                self.total = 0

            def update_counts_from_preds(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
                preds = y_pred.argmax(dim=1)
                self.correct += (preds == y_true).sum().item()
                self.total += y_true.size(0)

            def compute_metrics(self) -> Dict[str, float]:
                acc = self.correct / self.total if self.total > 0 else 0.0
                return {"acc": acc}

            def get_sync_values(self) -> List[float]:
                return [float(self.correct), float(self.total)]

            def load_sync_values(self, values: List[float]) -> None:
                self.correct, self.total = int(values[0]), int(values[1])

    See Also:
        - BinaryClassificationMetrics: Reference implementation for binary tasks
    """

    @abstractmethod
    def reset_and_initialize(self) -> None:
        """
        Reset all internal counters to zero.

        Called at the start of each epoch by the trainer. All accumulated
        counts from the previous epoch should be cleared.

        Example:
            def reset_and_initialize(self) -> None:
                self.tp = 0
                self.tn = 0
                self.fp = 0
                self.fn = 0
        """
        pass

    @abstractmethod
    def update_counts_from_preds(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Accumulate counts from one batch of predictions.

        Called after each forward pass. Should update internal counters
        based on model predictions and ground truth labels.

        Args:
            y_pred: Model output tensor. Shape depends on task:
                    - Binary: [batch_size] or [batch_size, 1] (logits)
                    - Multi-class: [batch_size, num_classes] (logits)
            y_true: Ground truth labels. Shape: [batch_size]
                    Values are class indices (0, 1, 2, ...).

        Note:
            - Apply sigmoid/softmax inside this method if needed
            - Use .item() when adding to counters to avoid memory leaks
            - Handle tensor shapes carefully (squeeze if needed)

        Example:
            def update_counts_from_preds(self, y_pred, y_true) -> None:
                probs = torch.sigmoid(y_pred.squeeze(-1))
                preds = (probs > self.threshold).int()
                self.tp += ((preds == 1) & (y_true == 1)).sum().item()
                self.fn += ((preds == 0) & (y_true == 1)).sum().item()
        """
        pass

    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated counts.

        Called at the end of each epoch after all batches have been processed.
        Returns a dictionary of metric names to values.

        Returns:
            Dict[str, float]: Metric names and their computed values.
                              Keys will be prefixed with "train_" or "test_"
                              by the trainer (e.g., "acc" -> "train_acc").

        Note:
            - Handle division by zero gracefully (return 0.0 or -1.0)
            - Keep metric names short but descriptive

        Example:
            def compute_metrics(self) -> Dict[str, float]:
                total = self.tp + self.tn + self.fp + self.fn
                acc = (self.tp + self.tn) / total if total > 0 else 0.0
                return {"acc": acc, "precision": self.tp / (self.tp + self.fp)}
        """
        pass

    @abstractmethod
    def get_sync_values(self) -> List[float]:
        """
        Export internal counters for DDP synchronization.

        Used by BaseTrainer._sync_metrics_ddp() to gather values from all
        ranks before calling torch.distributed.all_reduce().

        Returns:
            List[float]: All internal counters as floats in a fixed order.
                         Order must match load_sync_values().

        Example:
            def get_sync_values(self) -> List[float]:
                return [float(self.tp), float(self.tn),
                        float(self.fp), float(self.fn)]
        """
        pass

    @abstractmethod
    def load_sync_values(self, values: List[float]) -> None:
        """
        Restore internal counters after DDP all_reduce.

        Called after all_reduce to load the summed values from all ranks.
        The order of values matches get_sync_values().

        Args:
            values: Synced counter values (summed across all ranks).

        Example:
            def load_sync_values(self, values: List[float]) -> None:
                self.tp = int(values[0])
                self.tn = int(values[1])
                self.fp = int(values[2])
                self.fn = int(values[3])
        """
        pass
