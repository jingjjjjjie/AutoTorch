'''
Model architecture for the classifier.
'''
from torch import nn

class CustomClassifierModel(nn.Module):
    def __init__(self, backbone, head, freeze_backbone=False):
        super().__init__()
        self.feature_extractor = backbone
        self.mlp_head = head

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.mlp_head(self.feature_extractor(x))
