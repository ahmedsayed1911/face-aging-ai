import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import mediapipe as mp

# MediaPipe stable version
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
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


def detect_face_mediapipe(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    if not results.detections:
        raise ValueError("No face found")

    h, w, _ = image.shape
    bbox = results.detections[0].location_data.relative_bounding_box

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)

    return (y, x + width, y + height, x)


def process_image(your_model, image, video, source_age, target_age=0,
                  window_size=512, stride=256):

    if video:
        raise ValueError("Video processing not supported on Hugging Face")

    image = np.array(image)

    fl = detect_face_mediapipe(image)

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