from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64
import torch
from pathlib import Path
import uuid

from core.model_loader import load_model, load_masks
from core.inference import predict_image

# ===== Paths =====
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_WEIGHTS = PROJECT_ROOT / "best_unet_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ===== App =====
app = FastAPI(title="Face Aging API", version="3.0")

# serve images
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_masks(ASSETS_DIR)
model = load_model(MODEL_WEIGHTS, device)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    source_age: int = Form(...),
    target_age: int = Form(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict_image(model, image, source_age, target_age)

    # handle result
    if isinstance(result, str):
        if result.startswith("iVBOR") or len(result) > 1000:
            result = Image.open(io.BytesIO(base64.b64decode(result)))
        else:
            result = Image.open(result)

    # save image
    filename = f"{uuid.uuid4()}.png"
    filepath = OUTPUT_DIR / filename
    result.save(filepath)

    # return URL
    return {
        "image_url": f"/images/{filename}"
    }


# run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)