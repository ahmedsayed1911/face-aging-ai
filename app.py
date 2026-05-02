import gradio as gr
import torch
from PIL import Image
from pathlib import Path

from core.model_loader import load_model, load_masks
from core.inference import predict_image

# paths
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_WEIGHTS = PROJECT_ROOT / "best_unet_model.pth"

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_masks(ASSETS_DIR)
model = load_model(MODEL_WEIGHTS, device)

def inference_fn(image, source_age, target_age):
    result_path = predict_image(model, image, source_age, target_age)
    return Image.open(result_path)

demo = gr.Interface(
    fn=inference_fn,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(10, 90, value=25, label="Source Age"),
        gr.Slider(10, 90, value=60, label="Target Age"),
    ],
    outputs=gr.Image(type="pil"),
    title="Face Aging AI",
    description="Upload a face and change its age",
)

demo.launch()