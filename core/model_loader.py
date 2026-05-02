import torch
from pathlib import Path
from model.models import UNet
import scripts.test_functions as test_functions

def load_model(weights_path: Path, device: torch.device):
    """Load the UNet model and set to eval mode."""
    model = UNet().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_masks(assets_dir: Path):
    """Load mask images from assets directory and set in test_functions."""
    mask1024 = assets_dir / "mask1024.jpg"
    mask512 = assets_dir / "mask512.jpg"
    if not mask1024.exists() or not mask512.exists():
        raise FileNotFoundError(f"Mask files not found in {assets_dir}. Please place mask1024.jpg and mask512.jpg there.")
    test_functions.set_mask_paths(str(mask1024), str(mask512))