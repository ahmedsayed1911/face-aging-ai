from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import base64
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

app = FastAPI()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    source_age: int = Form(...),
    target_age: int = Form(...)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict_image(model, image, source_age, target_age)

    buffered = io.BytesIO()
    result.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "image_base64": img_str
    }