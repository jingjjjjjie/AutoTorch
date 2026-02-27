'''
Classification head builders.
'''
from torch import nn

def classification_head(cfg):
    """Build the MLP classification head from config."""
    hidden = cfg['model']['hidden_units']
    return nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )
