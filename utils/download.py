import os
import gdown
import torch
from torchvision.models import vgg19
from utils.settings import get_device

def check_dir(dir_name: str) -> bool:
    return os.path.isdir(dir_name)

def download_data(dir_name="assets") -> None:
    if not check_dir(dir_name): 
        os.mkdir(dir_name)
    os.chdir(dir_name)
    gdown.download(
        "https://drive.google.com/uc?id=1fcB4_jHa-wCr6Nlr81rCWQ5LvW3d27vR", quiet=False
    )
    gdown.download(
        "https://drive.google.com/uc?id=1dGnUAqymwYp-jLPqB7EgOdjsCvDY9y55", quiet=False
    )
    os.chdir("..")

def get_model(device: torch.device = get_device()):
    model = vgg19(weights='IMAGENET1K_V1').features.to(device)
    return model