import base64
import tempfile
from pathlib import Path
from PIL import Image
from scripts.test_functions import process_image, process_video

def predict_image(model, image: Image.Image, source_age: int, target_age: int, window_size=512, stride=256) -> str:
    """
    Age a single image. Returns base64 string of the result image.
    """
    result_pil = process_image(model, image, video=False, source_age=source_age,
                               target_age=target_age, window_size=window_size, stride=stride)
    # Convert PIL to base64
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        result_pil.save(tmp.name, format="PNG")
        tmp_path = Path(tmp.name)
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    tmp_path.unlink()
    return b64

def predict_video(model, video_path: str, source_age: int, target_age: int,
                  window_size=512, stride=256, frame_count=0) -> str:
    """
    Age an entire video. Returns base64 string of the output MP4.
    """
    out_path = process_video(model, video_path, source_age, target_age,
                             window_size=window_size, stride=stride, frame_count=frame_count)
    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # Clean up temp file
    Path(out_path).unlink()
    return b64