import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch.nn.functional as F

import sys
sys.path.append("../..")
from src.unet.model import UNet
from src.data.deer import get_deer_dataloaders
from src.unet.diffuser import Diffuser

class LightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.num_timesteps = 1000
        self.device_name = 'mps'
        self.diffuser = Diffuser(self.num_timesteps, device=self.device_name) 

    def training_step(self, batch, batch_idx):
        x, _= batch
        t = torch.randint(1, self.num_timesteps+1, (len(x),), device=self.device_name)
        x_noisy, noise = self.diffuser.add_noise(x, t)
        noise_pred = self.model(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        t = torch.randint(1, self.num_timesteps+1, (len(x),), device=self.device)
        x_noisy, noise = self.diffuser.add_noise(x, t)
        noise_pred = self.model(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)
        
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        # オプティマイザーの設定
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main():
    # Wandbの初期化
    wandb.init(project="shika")

    # データセットとデータローダーの準備
    train_data_loader, val_data_loader = get_deer_dataloaders(batch_size=128, num_workers=8)

    # モデルの初期化
    model = LightningModule()

    # Wandbロガーの設定
    wandb_logger = WandbLogger()

    # チェックポイントの設定
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )

    # トレーナーの設定と学習の実行
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, train_data_loader, val_data_loader)

    # Wandbの終了
    wandb.finish()

if __name__ == "__main__":
    main()