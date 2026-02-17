from torch import optim

OPTIMIZER_MAP = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
}
