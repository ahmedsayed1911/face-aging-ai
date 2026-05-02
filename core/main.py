from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
from pathlib import Path
import uvicorn

from app.model_loader import load_model, load_masks
from app.inference import predict_image

# ----- Project root detection -----
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_WEIGHTS = PROJECT_ROOT / "best_unet_model.pth"

# ----- FastAPI app -----
app = FastAPI(title="Face Re-Aging API", description="Age faces in images/videos using a UNet model", version="1.0")

# CORS: allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files (if you want to serve the HTML directly)
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ----- Global variables (set once at startup) -----
model = None
device = None

@app.on_event("startup")
async def startup_event():
    global model, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load masks first
    load_masks(ASSETS_DIR)
    print("Masks loaded successfully")

    # Load model
    if not MODEL_WEIGHTS.exists():
        raise RuntimeError(f"Model weights not found at {MODEL_WEIGHTS}. Please place best_unet_model.pth in the project root.")
    model = load_model(MODEL_WEIGHTS, device)
    print("Model loaded successfully")

@app.on_event("shutdown")
async def shutdown_event():
    # Optional cleanup
    pass

# ----- Root endpoint that serves the HTML frontend -----
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Frontend not found. Please ensure frontend/index.html exists.</h1>")

# ----- Image prediction endpoint -----
@app.post("/predict-image")
async def predict_image_endpoint(
    file: UploadFile = File(...),
    source_age: int = Form(..., ge=10, le=90),
    target_age: int = Form(..., ge=10, le=90),
):
    """
    Accepts an image file, current age, target age.
    Returns a JSON with 'image_base64' containing the PNG result.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        from PIL import Image
        import io
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        result_b64 = predict_image(model, image, source_age, target_age)
        return {"image_base64": result_b64}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)