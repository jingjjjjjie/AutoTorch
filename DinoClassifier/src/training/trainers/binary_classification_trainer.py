from typing import Tuple
import torch
from .base_trainer import BaseTrainer


class BinaryClassificationTrainer(BaseTrainer):
    """Trainer for binary classification tasks (id fraud)."""

    def forward_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)

        y_pred = self.model(X).squeeze(dim=1)  # [batch,1] -> [batch]
        loss = self.loss_fn(y_pred, y.float())

        return y_pred, y, loss