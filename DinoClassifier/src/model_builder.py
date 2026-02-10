import torch
from torch import nn

class CustomClassifierModel(nn.Module):
    def __init__(self, backbone_model, backbone_model_output_dim, freeze_backbone=False):
        super().__init__()
        self.feature_extractor = backbone_model
        self.mlp_head = nn.Sequential(
                        nn.LayerNorm(backbone_model_output_dim),
                        nn.Linear(backbone_model_output_dim, 128),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 1)
        )
        # Model: logits
        # Loss: BCEWithLogits
        #           └ sigmoid inside
    
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.mlp_head(self.feature_extractor(x))


if __name__== "__main__":
    # load the backbone model
    # reference: https://github.com/facebookresearch/dinov3
    device = torch.device("cuda:0")
    REPO_DIR = "/home/jingjie/dev/dino/dinov3"
    CHECKPOINT_PATH = "/home/jingjie/dev/dino/DinoClassifier/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    
    dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT_PATH)
    
    # instantiate a custom classifier model with the loaded backbone
    model = CustomClassifierModel(
                backbone_model=dinov3_vits16,
                backbone_model_output_dim=384,
                freeze_backbone=False
                ).to(device) # move to device

    model.eval() # set to eval mode

    # prepare dummy input to test forward pass
    # batchsize of 2, 3 rgb channels, image is 224*224
    dummy_input = torch.randn(2, 3, 512, 512).to(device) # also move to device, ensure same location with model

    # run forward pass and visualize the outputs
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.sigmoid(output)

    print("Output shape:", output.shape)
    print("Output:", output)
    print(probs)