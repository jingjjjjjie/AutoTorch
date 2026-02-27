"""
Task-agnostic trainer. Subclass and implement forward_step() for your task.
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from utils.device import is_main_process


class BaseTrainer(ABC):
    """Abstract base trainer that supports most supervised learning tasks
      Subclass and implement forward_step() for your task."""

    def __init__(self, model, loss_fn, optimizer, metrics_handler, device,
                scheduler=None, sampler=None, early_stopping=None, checkpoint=None):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics_handler = metrics_handler
        self.device = device
        self.scheduler = scheduler
        self.sampler = sampler
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint

    @abstractmethod
    def forward_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one batch. Must be implemented by subclass.

        Args:
            batch: Whatever your dataloader yields (e.g., (X, y) or (images, targets, lengths))

        Returns:
            Tuple of (y_pred, y_true, loss):
                - y_pred: Model predictions (for metrics)
                - y_true: Ground truth labels (for metrics)
                - loss: Computed loss (for backward pass)
        """
        pass

    def _sync_metrics_ddp(self) -> None:
        """Sync metrics across DDP ranks using all_reduce.
        Sums metric counts from all GPUs so each rank
        has the global totals. Skips if DDP is not initialized.
        """
        if not torch.distributed.is_initialized():
            return

        # Extract counts from metrics handler (e.g., [tp, tn, fp, fn])
        sync_values = self.metrics_handler.get_sync_values()

        # Convert to tensor on GPU (needed for all_reduce communication)
        tensor = torch.tensor(sync_values, dtype=torch.float64, device=self.device)

        # Sum values across all ranks (each GPU adds its counts to the tensor)
        torch.distributed.all_reduce(tensor)

        # Load summed values back into metrics handler
        self.metrics_handler.load_sync_values(tensor.tolist())

    def _train_step(self, dataloader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.metrics_handler.reset_and_initialize()
        loss_sum = 0.0

        for batch in dataloader:
            y_pred, y_true, loss = self.forward_step(batch)  # subclass handles this

            self.optimizer.zero_grad()  # clear gradients
            loss.backward()             # compute gradients
            self.optimizer.step()       # update weights

            loss_sum += loss.item()
            self.metrics_handler.update_counts_from_preds(y_pred, y_true)

        results = {"loss": loss_sum / len(dataloader)}
        results.update(self.metrics_handler.compute_metrics())
        return results

    def _eval_step(self, dataloader) -> Dict[str, float]:
        """Run one evaluation epoch."""
        self.model.eval()
        self.metrics_handler.reset_and_initialize()
        loss_sum = 0.0
        num_batches = len(dataloader)

        with torch.inference_mode():
            for batch in dataloader:
                y_pred, y_true, loss = self.forward_step(batch)
                loss_sum += loss.item()
                self.metrics_handler.update_counts_from_preds(y_pred, y_true)

        # Sync across DDP ranks
        if torch.distributed.is_initialized():
            loss_tensor = torch.tensor([loss_sum, num_batches], dtype=torch.float64, device=self.device)
            torch.distributed.all_reduce(loss_tensor)
            loss_sum, num_batches = loss_tensor[0].item(), loss_tensor[1].item()
            self._sync_metrics_ddp()

        results = {"loss": loss_sum / num_batches}
        results.update(self.metrics_handler.compute_metrics())
        return results

    def train(self, train_dataloader, test_dataloader, epochs) -> Dict[str, List]:
        """Full training loop."""
        history = {"lr": []}

        for epoch in tqdm(range(epochs), disable=not is_main_process()):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            train_results_dict = self._train_step(train_dataloader)
            test_results_dict = self._eval_step(test_dataloader)

            if self.scheduler:
                self.scheduler.step()

            # Track Learning Rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history["lr"].append(current_lr)

            # Store results with prefixes
            for key, val in train_results_dict.items():
                history.setdefault(f"train_{key}", []).append(val)
            for key, val in test_results_dict.items():
                history.setdefault(f"test_{key}", []).append(val)

            # Print progress
            if is_main_process():
                parts = [f"Epoch: {epoch+1}"]
                for key, val in train_results_dict.items():
                    parts.append(f"train_{key}: {val:.4f}")
                for key, val in test_results_dict.items():
                    parts.append(f"test_{key}: {val:.4f}")
                print(" | ".join(parts))

            # Checkpoint
            if self.checkpoint is not None and is_main_process():
                self.checkpoint.save(self.model, self.optimizer, epoch, test_results_dict["loss"])

            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping.check_early_stop(test_results_dict["loss"])
                if self.early_stopping.stop_training:
                    if is_main_process():
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        return history

