'''
Loss function registry.
'''
from torch import nn

LOSS_FN_MAP = {
    'cross_entropy': nn.CrossEntropyLoss,
    'bce': nn.BCELoss,
    'bce_with_logits': nn.BCEWithLogitsLoss,
    'mse': nn.MSELoss,
}
