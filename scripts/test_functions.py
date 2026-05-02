import face_recognition
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.io import write_video
import tempfile
import subprocess
import json
from ffmpy import FFmpeg, FFprobe
from PIL import Image
import sys

# These will be set by the application on startup using absolute paths
mask_file = None
small_mask_file = None

def set_mask_paths(mask1024_path, mask512_path):
    """Set the global mask tensors from given absolute file paths."""
    global mask_file, small_mask_file
    mask_file = torch.from_numpy(np.array(Image.open(mask1024_path).convert('L'))) / 255
    small_mask_file = torch.from_numpy(np.array(Image.open(mask512_path).convert('L'))) / 255

def sliding_window_tensor(input_tensor, window_size, stride, your_model):
    """
    Apply aging operation on input tensor using a sliding-window method.
    """
    global mask_file, small_mask_file
    if mask_file is None or small_mask_file is None:
        raise RuntimeError("Mask tensors not initialized. Call set_mask_paths first.")

    input_tensor = input_tensor.to(next(your_model.parameters()).device)
    mask = mask_file.to(next(your_model.parameters()).device)
    small_mask = small_mask_file.to(next(your_model.parameters()).device)

    n, c, h, w = input_tensor.size()
    output_tensor = torch.zeros((n, 3, h, w), dtype=input_tensor.dtype, device=input_tensor.device)
    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32, device=input_tensor.device)

    add = 2 if window_size % stride != 0 else 1

    for y in range(0, h - window_size + add, stride):
        for x in range(0, w - window_size + add, stride):
            window = input_tensor[:, :, y:y + window_size, x:x + window_size]
            input_variable = Variable(window, requires_grad=False)
            with torch.no_grad():
                output = your_model(input_variable)
            output_tensor[:, :, y:y + window_size, x:x + window_size] += output * small_mask
            count_tensor[:, :, y:y + window_size, x:x + window_size] += small_mask

    count_tensor = torch.clamp(count_tensor, min=1.0)
    output_tensor /= count_tensor
    output_tensor *= mask
    return output_tensor.cpu()


def process_image(your_model, image, video, source_age, target_age=0,
                  window_size=512, stride=256, steps=18):
    """
    Age the person in the image.
    If video=False, return a PIL image.
    If video=True, return the path to an MP4 video.
    """
    if video:
        target_age = 0
    input_size = (1024, 1024)

    image = np.array(image)
    if video:
        height, width, depth = image.shape
        new_width = width if width % 2 == 0 else width - 1
        new_height = height if height % 2 == 0 else height - 1
        image = image[:new_height, :new_width, :]

    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        raise ValueError("No face found in the image.")
    fl = face_locations[0]

    margin_y_t = int((fl[2] - fl[0]) * .63 * .85)
    margin_y_b = int((fl[2] - fl[0]) * .37 * .85)
    margin_x = int((fl[1] - fl[3]) // (2 / .85))
    margin_y_t += 2 * margin_x - margin_y_t - margin_y_b

    l_y = max([fl[0] - margin_y_t, 0])
    r_y = min([fl[2] + margin_y_b, image.shape[0]])
    l_x = max([fl[3] - margin_x, 0])
    r_x = min([fl[1] + margin_x, image.shape[1]])

    cropped_image = image[l_y:r_y, l_x:r_x, :]
    orig_size = cropped_image.shape[:2]

    cropped_image = transforms.ToTensor()(cropped_image)
    cropped_image_resized = transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(cropped_image)

    source_age_channel = torch.full_like(cropped_image_resized[:1, :, :], source_age / 100)
    target_age_channel = torch.full_like(cropped_image_resized[:1, :, :], target_age / 100)
    input_tensor = torch.cat([cropped_image_resized, source_age_channel, target_age_channel], dim=0).unsqueeze(0)

    image_tensor = transforms.ToTensor()(image)

    if video:
        interval = .8 / steps
        aged_cropped_images = torch.zeros((steps, 3, input_size[1], input_size[0]))
        for i in range(steps):
            input_tensor[:, -1, :, :] += interval
            aged_cropped_images[i, ...] = sliding_window_tensor(input_tensor, window_size, stride, your_model)

        aged_cropped_images_resized = transforms.Resize(orig_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(aged_cropped_images)
        image_tensor = image_tensor.repeat(steps, 1, 1, 1)
        image_tensor[:, :, l_y:r_y, l_x:r_x] += aged_cropped_images_resized
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_tensor = (image_tensor * 255).to(torch.uint8)

        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        write_video(output_file.name, image_tensor.permute(0, 2, 3, 1), 2)
        return output_file.name
    else:
        aged_cropped_image = sliding_window_tensor(input_tensor, window_size, stride, your_model)
        aged_cropped_image_resized = transforms.Resize(orig_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(aged_cropped_image)
        image_tensor[:, l_y:r_y, l_x:r_x] += aged_cropped_image_resized.squeeze(0)
        image_tensor = torch.clamp(image_tensor, 0, 1)
        return transforms.functional.to_pil_image(image_tensor)


def process_video(your_model, video_path, source_age, target_age, window_size=512, stride=256, frame_count=0):
    """
    Age every frame of a video. Returns path to output MP4.
    """
    frames_dir = tempfile.TemporaryDirectory()
    output_template = os.path.join(frames_dir.name, '%04d.jpg')

    if frame_count:
        ff = FFmpeg(inputs={video_path: None}, outputs={output_template: ['-vf', f'select=lt(n\\,{frame_count})', '-q:v', '1']})
    else:
        ff = FFmpeg(inputs={video_path: None}, outputs={output_template: ['-q:v', '1']})
    ff.run()

    ff_probe = FFprobe(inputs={video_path: None}, global_options=['-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1'])
    stdout, _ = ff_probe.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_rate = eval(stdout.decode('utf-8').strip())

    processed_dir = tempfile.TemporaryDirectory()

    for name in sorted(os.listdir(frames_dir.name)):
        image_path = os.path.join(frames_dir.name, name)
        image = Image.open(image_path).convert('RGB')
        image_aged = process_image(your_model, image, False, source_age, target_age, window_size, stride)
        image_aged.save(os.path.join(processed_dir.name, name))

    input_template = os.path.join(processed_dir.name, '%04d.jpg')
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ff_out = FFmpeg(inputs={input_template: f'-framerate {frame_rate}'}, global_options=['-y'], outputs={output_file.name: ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']})
    ff_out.run()

    frames_dir.cleanup()
    processed_dir.cleanup()
    return output_file.name