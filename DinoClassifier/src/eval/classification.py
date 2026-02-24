'''
Stores the classification metrics for binary classification: accuracy, APCER, BPCER and ACER 
Use the evaluate function: run_eval_classification to evaluate
'''
import torch

def count_tp_tn_fp_fn(logits, y_true):
    """Returns raw TP, TN, FP, FN counts for a batch."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    tp = ((preds == 1) & (y_true == 1)).sum().item()
    tn = ((preds == 0) & (y_true == 0)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()

    return tp, tn, fp, fn

def compute_binary_metrics(tp, tn, fp, fn):
    """Computes acc, apcer, bpcer from accumulated counts.
    Returns -1 for apcer/bpcer if that class is missing from the dataset.
    """
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    apcer = fn / (tp + fn) if (tp + fn) else -1  # -1 if no attack/fraud samples
    bpcer = fp / (tn + fp) if (tn + fp) else -1  # -1 if no genuine samples
    return acc, apcer, bpcer

def run_eval_classification(model, dataloader, device, threshold=0.5):
    """Run evaluation and return metrics."""
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(dim=1)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()

            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy, apcer, bpcer = compute_binary_metrics(tp, tn, fp, fn)
    acer = (apcer + bpcer) / 2

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy, "apcer": apcer, "bpcer": bpcer, "acer": acer,
    }, all_preds, all_probs