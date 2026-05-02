import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

# تحميل face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

mask_file = None
small_mask_file = None


def set_mask_paths(mask1024_path, mask512_path):
    global mask_file, small_mask_file
    mask_file = torch.from_numpy(np.array(Image.open(mask1024_path).convert('L'))) / 255
    small_mask_file = torch.from_numpy(np.array(Image.open(mask512_path).convert('L'))) / 255


def sliding_window_tensor(input_tensor, window_size, stride, your_model):
    global mask_file, small_mask_file

    input_tensor = input_tensor.to(next(your_model.parameters()).device)
    mask = mask_file.to(next(your_model.parameters()).device)
    small_mask = small_mask_file.to(next(your_model.parameters()).device)

    n, c, h, w = input_tensor.size()
    output_tensor = torch.zeros((n, 3, h, w), dtype=input_tensor.dtype, device=input_tensor.device)
    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32, device=input_tensor.device)

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window = input_tensor[:, :, y:y + window_size, x:x + window_size]
            with torch.no_grad():
                output = your_model(window)
            output_tensor[:, :, y:y + window_size, x:x + window_size] += output * small_mask
            count_tensor[:, :, y:y + window_size, x:x + window_size] += small_mask

    count_tensor = torch.clamp(count_tensor, min=1.0)
    output_tensor /= count_tensor
    output_tensor *= mask
    return output_tensor.cpu()


# ✅ Face detection بـ OpenCV
def detect_face_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        raise ValueError("No face found")

    x, y, w, h = faces[0]

    return (y, x + w, y + h, x)


def process_image(your_model, image, video, source_age, target_age=0,
                  window_size=512, stride=256):

    if video:
        raise ValueError("Video not supported on Hugging Face")

    image = np.array(image)

    # 🔥 detection هنا
    fl = detect_face_opencv(image)

    margin_y_t = int((fl[2] - fl[0]) * 0.6)
    margin_y_b = int((fl[2] - fl[0]) * 0.4)
    margin_x = int((fl[1] - fl[3]) * 0.5)

    l_y = max(fl[0] - margin_y_t, 0)
    r_y = min(fl[2] + margin_y_b, image.shape[0])
    l_x = max(fl[3] - margin_x, 0)
    r_x = min(fl[1] + margin_x, image.shape[1])

    cropped = image[l_y:r_y, l_x:r_x]
    orig_size = cropped.shape[:2]

    cropped = transforms.ToTensor()(cropped)
    cropped = transforms.Resize((1024, 1024))(cropped)

    source_age_channel = torch.full_like(cropped[:1], source_age / 100)
    target_age_channel = torch.full_like(cropped[:1], target_age / 100)

    input_tensor = torch.cat(
        [cropped, source_age_channel, target_age_channel],
        dim=0
    ).unsqueeze(0)

    image_tensor = transforms.ToTensor()(image)

    aged = sliding_window_tensor(input_tensor, window_size, stride, your_model)
    aged_resized = transforms.Resize(orig_size)(aged)

    image_tensor[:, l_y:r_y, l_x:r_x] += aged_resized.squeeze(0)
    image_tensor = torch.clamp(image_tensor, 0, 1)

    return transforms.functional.to_pil_image(image_tensor)


def process_video(*args, **kwargs):
    raise ValueError("Video processing not supported on Hugging Face Spaces")