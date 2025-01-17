import torch
from torch import nn

def _pos_encoding(t, output_dim, device='cpu'):
    D = output_dim
    v = torch.zeros(D, device=device)
    
    i = torch.arange(0, D, device=device)
    div_term = 10000 ** (i/D)

    v[0::2] = torch.sin(t/div_term[0::2])
    v[1::2] = torch.cos(t/div_term[1::2])
    return v

def pos_encoding(ts, output_dim, device='cpu'):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device = device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
    return v

class UNet(nn.Module):
    def __init__(self, in_ch=3, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")


    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        return  self.out(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, 3, padding=1),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(),
              nn.Conv2d(out_ch, out_ch, 3, padding=1),
              nn.BatchNorm2d(out_ch),
              nn.ReLU()
            )
        self.mlp = nn.Sequential(
              nn.Linear(time_embed_dim, in_ch),
              nn.ReLU(),
              nn.Linear(in_ch, in_ch)
            )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y

if __name__ == "__main__":
    unet = UNet()
    random_tensor = torch.rand(5, 1, 128, 128)
    t = torch.randint(1, 5, (len(random_tensor),))
    y = unet(random_tensor, t)

