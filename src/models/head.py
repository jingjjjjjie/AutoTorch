'''
Classification head architectures.
'''
from torch import nn


class ClassificationHeadV1(nn.Module):
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