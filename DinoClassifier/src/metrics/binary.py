'''
Binary classification metrics: accuracy, APCER, BPCER.
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
    """Computes acc, apcer, bpcer from accumulated counts."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    apcer = fn / (tp + fn) if (tp + fn) else 0
    bpcer = fp / (tn + fp) if (tn + fp) else 0
    return acc, apcer, bpcer
