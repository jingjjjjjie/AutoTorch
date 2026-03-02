'''
Classification head builders.
'''
from torch import nn

class ClassificationHeadV1(nn.Module):
    """Simple head: LayerNorm -> Linear -> GELU -> Dropout -> Linear"""
    def __init__(self, hidden):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(x)


class ClassificationHeadV2(nn.Module):
    """Deeper head: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear"""
    def __init__(self, hidden):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(x)


class ResidualHeadV1(nn.Module):
    """Residual head: LayerNorm -> residual block -> Linear"""
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.norm(x)
        x = x + self.block(x)
        return self.out(x)

HEAD_MAP = {
    'v1': ClassificationHeadV1,
    'v2': ClassificationHeadV2,
    'residual_v1': ResidualHeadV1,
}



def build_head(cfg):
    """Create classification head from config."""
    head_type = cfg['model'].get('head_type', 'v1')
    hidden = cfg['model']['hidden_units']

    if head_type not in HEAD_MAP:
        raise ValueError(f"Unknown head_type '{head_type}'. Available: {list(HEAD_MAP.keys())}")

    return HEAD_MAP[head_type](hidden)