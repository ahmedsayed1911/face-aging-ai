import gradio as gr
import torch
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
    # 🔥 بيرجع PIL Image مباشرة
    result = predict_image(model, image, source_age, target_age)
    return result


demo = gr.Interface(
    fn=inference_fn,
    inputs=[
        gr.Image(type="pil", label="Upload Face Image"),
        gr.Slider(10, 90, value=25, label="Source Age"),
        gr.Slider(10, 90, value=60, label="Target Age"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Face Aging AI",
    description="Upload a face image and change its age using AI",
)

demo.launch()