'''
Model architecture for the classifier.
'''
from torch import nn

class CustomClassifierModel(nn.Module):
    def __init__(self, backbone, output_dim, freeze=False):
        super().__init__()
        self.feature_extractor = backbone # the feature extractor backbone
        self.mlp_head = nn.Sequential( 
                        nn.LayerNorm(output_dim), # output dim is the backbone model's output dim
                        nn.Linear(output_dim, 128),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 1)
        )

        if freeze: # freeze backbone
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.mlp_head(self.feature_extractor(x))
