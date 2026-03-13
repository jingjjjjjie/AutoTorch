"""
Contains functions for training and validating a PyTorch model.
"""
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
from utils.device import is_main_process
from eval.idfraud.metrics import count_tp_tn_fp_fn, compute_binary_metrics


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               non_blocking: bool = True) -> Tuple[float, float, float, float]:
    """
    Trains a PyTorch model for a single epoch.
    
    Returns:
    A tuple of (train_loss, train_acc, train_apcer, train_bpcer).
    """
    # set the model to train mode
    model.train()

    # initial values to be recorded
    train_loss = 0
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device, non_blocking=non_blocking), \
               y.to(device, non_blocking=non_blocking)

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

def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              non_blocking: bool = True) -> Tuple[float, float, float, float]:
    """
    Validates a PyTorch model for a single epoch.

    Returns:
    A tuple of (val_loss, val_acc, val_apcer, val_bpcer).
    """
    
    model.eval()
    val_loss_sum = 0
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=non_blocking), \
                   y.to(device, non_blocking=non_blocking)

            y_pred = model(X).squeeze(dim=1)  # [batch, 1] -> [batch]
            loss = loss_fn(y_pred, y.float())
            val_loss_sum += loss.item()

            tp, tn, fp, fn = count_tp_tn_fp_fn(y_pred, y)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    # Aggregate metrics across all ranks for DDP
    if torch.distributed.is_initialized():
        stats = torch.tensor(
            [val_loss_sum, len(dataloader), total_tp, total_tn, total_fp, total_fn],
            dtype=torch.float64, device=device
        )
        torch.distributed.all_reduce(stats)
        val_loss = stats[0].item() / stats[1].item()
        total_tp, total_tn, total_fp, total_fn = [int(x) for x in stats[2:].tolist()]
    else:
        val_loss = val_loss_sum / len(dataloader)

    acc, apcer, bpcer = compute_binary_metrics(total_tp, total_tn, total_fp, total_fn)
    return val_loss, acc, apcer, bpcer

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler=None,
          sampler=None,
          early_stopping=None,
          checkpoint=None,
          on_epoch_end=None) -> Dict[str, List]:
    """
    Trains and validates a PyTorch model.

    Returns:
    A dictionary with train/val loss, acc, apcer, bpcer per epoch.
    """
    
    results = {
        "train_loss": [], "train_acc": [], "train_apcer": [], "train_bpcer": [],
        "val_loss": [], "val_acc": [], "val_apcer": [], "val_bpcer": [],
        "lr": []
    }

    model.to(device)

    for epoch in tqdm(range(epochs), disable=not is_main_process()):
        # Set epoch for distributed sampler (ensures proper shuffling)
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_loss, train_acc, train_apcer, train_bpcer = train_step(
            model=model, 
            dataloader=train_dataloader,
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device
        )
        val_loss, val_acc, val_apcer, val_bpcer = val_step(
            model=model, 
            dataloader=val_dataloader,
            loss_fn=loss_fn, 
            device=device
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
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f} | "
                f"val_apcer: {val_apcer:.4f} | "
                f"val_bpcer: {val_bpcer:.4f}"
            )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_apcer"].append(train_apcer)
        results["train_bpcer"].append(train_bpcer)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_apcer"].append(val_apcer)
        results["val_bpcer"].append(val_bpcer)

        # Save checkpoint (main process only)
        if checkpoint is not None and is_main_process():
            checkpoint.save(model, optimizer, epoch, val_loss)

        # Check early stopping condition
        if early_stopping is not None:
            early_stopping.check_early_stop(val_loss)
            if early_stopping.stop_training:
                break

        # Per-epoch callback (e.g., save logs/plots)
        if on_epoch_end is not None:
            on_epoch_end(results)

    return results