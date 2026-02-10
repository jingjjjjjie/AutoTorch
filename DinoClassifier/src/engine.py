"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def color(val, curr, prev, lower_is_better=False):
    s = f"{val:.4f}"
    if prev is None: return s
    if lower_is_better:
        if curr < prev: return f"{GREEN}{s}{RESET}"
        if curr > prev: return f"{RED}{s}{RESET}"
    else:
        if curr > prev: return f"{GREEN}{s}{RESET}"
        if curr < prev: return f"{RED}{s}{RESET}"
    return s

def binary_counts(logits, y_true):
    """Returns raw TP, TN, FP, FN counts for a batch."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    tp = ((preds == 1) & (y_true == 1)).sum().item()
    tn = ((preds == 0) & (y_true == 0)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()

    return tp, tn, fp, fn


def compute_metrics(tp, tn, fp, fn):
    """Computes acc, apcer, bpcer from accumulated counts."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    apcer = fn / (tp + fn) if (tp + fn) else 0
    bpcer = fp / (tn + fp) if (tn + fp) else 0
    return acc, apcer, bpcer

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

        tp, tn, fp, fn = binary_counts(y_pred, y)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    train_loss = train_loss / len(dataloader)
    acc, apcer, bpcer = compute_metrics(total_tp, total_tn, total_fp, total_fn)
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

            tp, tn, fp, fn = binary_counts(y_pred, y)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    test_loss = test_loss / len(dataloader)
    acc, apcer, bpcer = compute_metrics(total_tp, total_tn, total_fp, total_fn)
    return test_loss, acc, apcer, bpcer

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Returns:
    A dictionary with train/test loss, acc, apcer, bpcer per epoch.
    """
    results = {
        "train_loss": [], "train_acc": [], "train_apcer": [], "train_bpcer": [],
        "test_loss": [], "test_acc": [], "test_apcer": [], "test_bpcer": []
    }

    model.to(device)
    prev = {}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_apcer, train_bpcer = train_step(
            model=model, dataloader=train_dataloader,
            loss_fn=loss_fn, optimizer=optimizer, device=device
        )
        test_loss, test_acc, test_apcer, test_bpcer = test_step(
            model=model, dataloader=test_dataloader,
            loss_fn=loss_fn, device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {color(train_loss, train_loss, prev.get('tl'), True)} | "
            f"train_acc: {color(train_acc, train_acc, prev.get('ta'))} | "
            f"train_apcer: {color(train_apcer, train_apcer, prev.get('tap'), True)} | "
            f"train_bpcer: {color(train_bpcer, train_bpcer, prev.get('tbp'), True)} | "
            f"test_loss: {color(test_loss, test_loss, prev.get('vl'), True)} | "
            f"test_acc: {color(test_acc, test_acc, prev.get('va'))} | "
            f"test_apcer: {color(test_apcer, test_apcer, prev.get('vap'), True)} | "
            f"test_bpcer: {color(test_bpcer, test_bpcer, prev.get('vbp'), True)}"
        )

        prev = {'tl': train_loss, 'ta': train_acc, 'tap': train_apcer, 'tbp': train_bpcer,
                'vl': test_loss, 'va': test_acc, 'vap': test_apcer, 'vbp': test_bpcer}
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_apcer"].append(train_apcer)
        results["train_bpcer"].append(train_bpcer)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_apcer"].append(test_apcer)
        results["test_bpcer"].append(test_bpcer)

    return results