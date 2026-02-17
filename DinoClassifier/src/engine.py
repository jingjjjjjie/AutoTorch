"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from utils.device import is_main_process
from metrics import count_tp_tn_fp_fn, compute_binary_metrics
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, float, float]:
    """Trains a PyTorch model for a single epoch.

    Returns:
    A tuple of (train_loss, train_acc, train_apcer, train_bpcer).
    """
    model.train()
    train_loss = 0
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X).squeeze(dim=1)  # [batch, 1] -> [batch]
        loss = loss_fn(y_pred, y.float())
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tp, tn, fp, fn = count_tp_tn_fp_fn(y_pred, y)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    train_loss = train_loss / len(dataloader)
    acc, apcer, bpcer = compute_binary_metrics(total_tp, total_tn, total_fp, total_fn)
    return train_loss, acc, apcer, bpcer

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, float, float]:
    """Tests a PyTorch model for a single epoch.

    Returns:
    A tuple of (test_loss, test_acc, test_apcer, test_bpcer).
    """
    model.eval()
    test_loss = 0
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X).squeeze(dim=1)  # [batch, 1] -> [batch]
            loss = loss_fn(y_pred, y.float())
            test_loss += loss.item()

            tp, tn, fp, fn = count_tp_tn_fp_fn(y_pred, y)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    test_loss = test_loss / len(dataloader)
    acc, apcer, bpcer = compute_binary_metrics(total_tp, total_tn, total_fp, total_fn)
    return test_loss, acc, apcer, bpcer

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler=None,
          sampler=None,
          early_stopping=None,
          checkpoint=None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Args:
        sampler: DistributedSampler for DDP (calls set_epoch each epoch)

    Returns:
    A dictionary with train/test loss, acc, apcer, bpcer per epoch.
    """
    

    results = {
        "train_loss": [], "train_acc": [], "train_apcer": [], "train_bpcer": [],
        "test_loss": [], "test_acc": [], "test_apcer": [], "test_bpcer": [],
        "lr": []
    }

    model.to(device)
    prev = {}

    for epoch in tqdm(range(epochs), disable=not is_main_process()):
        # Set epoch for distributed sampler (ensures proper shuffling)
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_loss, train_acc, train_apcer, train_bpcer = train_step(
            model=model, dataloader=train_dataloader,
            loss_fn=loss_fn, optimizer=optimizer, device=device
        )
        test_loss, test_acc, test_apcer, test_bpcer = test_step(
            model=model, dataloader=test_dataloader,
            loss_fn=loss_fn, device=device
        )

        if scheduler:
            scheduler.step()

        # Track current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        results["lr"].append(current_lr)

        if is_main_process():
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"train_apcer: {train_apcer:.4f} | "
                f"train_bpcer: {train_bpcer:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f} | "
                f"test_apcer: {test_apcer:.4f} | "
                f"test_bpcer: {test_bpcer:.4f}"
            )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_apcer"].append(train_apcer)
        results["train_bpcer"].append(train_bpcer)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_apcer"].append(test_apcer)
        results["test_bpcer"].append(test_bpcer)

        # Save checkpoint (main process only)
        if checkpoint is not None and is_main_process():
            checkpoint.save(model, optimizer, epoch, test_loss)

        # Check early stopping condition
        if early_stopping is not None:
            early_stopping.check_early_stop(test_loss)
            if early_stopping.stop_training:
                if is_main_process():
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    return results