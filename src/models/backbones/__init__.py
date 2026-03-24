'''
Backbone model loaders and registry.
'''
from .dino import load_dino_model, WEIGHTS_MAP as DINO_MODELS
from .van import load_van_model, WEIGHTS_MAP as VAN_MODELS
from .unireplknet import load_unireplknet_model, WEIGHTS_MAP as UNIREPLKNET_MODELS
from .overlock import load_overlock_model, WEIGHTS_MAP as OVERLOCK_MODELS

# Model router: maps model name to loader function
BACKBONE_LOADERS = {
    **{name: load_dino_model for name in DINO_MODELS},
    **{name: load_van_model for name in VAN_MODELS},
    **{name: load_unireplknet_model for name in UNIREPLKNET_MODELS},
    **{name: load_overlock_model for name in OVERLOCK_MODELS},
}

def load_backbone(model_name: str) -> tuple:
    """Load a backbone model by name, raises ValueError: If model_name is not recognized.
    Args:
        model_name: Specific name of the model (e.g., 'dinov3_vitb16', 'van_small')
    Returns:
        (model, output_dim) tuple
    """
    if model_name not in BACKBONE_LOADERS:
        available = list(BACKBONE_LOADERS.keys())
        raise ValueError(f"Unknown model: '{model_name}'. Available: {available}")

    loader = BACKBONE_LOADERS[model_name]
    return loader(model_name)


