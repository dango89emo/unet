import sys
sys.path.append("../..")

import torch
from src.unet.model import UNet
from src.unet.diffuser import Diffuser
from src.unet.train_unet import show_images

num_timesteps = 1000
device = "mps"
diffuser = Diffuser(num_timesteps, device=device)

unet = UNet(3)
unet.to(device)

a = {}
for v, k in torch.load('checkpoints/model-epoch=09-val_loss=0.06.ckpt')['state_dict'].items():
    a[v.replace('model.', '')] = k
unet.load_state_dict(a)

images = diffuser.sample(unet)
show_images(images)
