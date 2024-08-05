import sys
sys.path.append("../..")

import torch
from src.unet.model import UNet
from src.unet.diffuser import Diffuser
from src.unet.train_unet import show_images

num_timesteps = 1000
device = "mps"
diffuser = Diffuser(num_timesteps, device=device)

unet = UNet()
unet.to(device)
unet.load_state_dict(torch.load('model_weights.pth'))

images = diffuser.sample(unet)
show_images(images)
