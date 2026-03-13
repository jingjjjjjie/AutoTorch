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
    
class LegacyIDFraudHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, 128),
            nn.GELU(),

            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.head(x)