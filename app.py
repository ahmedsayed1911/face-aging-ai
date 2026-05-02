from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import torch
from pathlib import Path

from core.model_loader import load_model, load_masks
from core.inference import predict_image

# ===== Paths =====
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_WEIGHTS = PROJECT_ROOT / "best_unet_model.pth"

# ===== App =====
app = FastAPI(title="Face Aging API", version="1.0")

# ===== CORS (للموبايل) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load Model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_masks(ASSETS_DIR)
model = load_model(MODEL_WEIGHTS, device)


@app.get("/")
def root():
    return {"status": "ok", "message": "Face Aging API is running 🚀"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    source_age: int = Form(...),
    target_age: int = Form(...)
):
    # ===== Validate image =====
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # ===== Inference =====
    try:
        result = predict_image(model, image, source_age, target_age)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # ===== 🔥 Handle all cases =====
    if isinstance(result, str):

        # 🟢 case: base64 string
        if result.startswith("iVBOR") or len(result) > 1000:
            try:
                result = Image.open(io.BytesIO(base64.b64decode(result)))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Base64 decode failed: {e}")

        # 🟢 case: file path
        else:
            try:
                result = Image.open(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Image path invalid: {e}")

    # 🟢 ensure it's PIL Image
    if not isinstance(result, Image.Image):
        raise HTTPException(status_code=500, detail="Output is not a valid image")

    # ===== Convert to base64 =====
    buffered = io.BytesIO()
    result.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image_base64": img_str}


# ===== Run (HF Spaces يحتاج ده) =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)