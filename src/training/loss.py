'''
Loss function registry.
'''
from torch import nn

LOSS_FN_MAP = {
    'bce_with_logits': nn.BCEWithLogitsLoss,
}
