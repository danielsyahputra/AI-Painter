import io
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from utils.settings import get_device

DEVICE = get_device()

def get_transform(pair_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(pair_size), 
        transforms.ToTensor()])
    return transform

def load_style_content(content_uploaded_file, style_uploaded_file, 
                        transform: transforms.Compose = get_transform, 
                        device: torch.device = DEVICE):
    content_bytes = content_uploaded_file.getvalue()
    content = np.array(Image.open(io.BytesIO(content_bytes)))
    style_bytes = style_uploaded_file.getvalue()
    style = np.array(Image.open(io.BytesIO(style_bytes)))
    transform = get_transform((300, 300))
    style_image = transform(style).unsqueeze(0)
    content_image = transform(content).unsqueeze(0)
    return style_image.to(device), content_image.to(device)

def get_unloader():
    return transforms.ToPILImage()