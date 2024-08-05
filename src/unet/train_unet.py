import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append("../..")
from src.unet.model import UNet
from src.unet.diffuser import Diffuser


def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.show()

def main():
    batch_size = 128
    num_timesteps = 1000
    epochs = 10
    lr = 1e-3
    device = 'mps'

    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    diffuser = Diffuser(num_timesteps, device=device)
    model = UNet()
    model.to(device)
    optimizer=Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        for images, _ in tqdm(dataloader):
            optimizer.zero_grad()
            x = images.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Loss: {loss_avg}')

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__=="__main__":
    main()
