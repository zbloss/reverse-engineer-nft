import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from .psnr_loss import PSNRLoss
from .ssim_loss import SSIMLoss
from .nft_reverse_engineer_stylegan import NFTReverseEngineerStyleGAN


class NFTReverseEngineer(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, target_loss: str = "ssim", use_cuda: bool = False):
        super().__init__()
        self.learning_rate = learning_rate
        self.target_loss = target_loss
        self.use_cuda = use_cuda

        self.mse_loss = nn.MSELoss()
        self.psnr_loss = PSNRLoss()
        self.ssim_loss = SSIMLoss()
        self.model = NFTReverseEngineerStyleGAN(
            target_loss=self.target_loss, 
            use_cuda=self.use_cuda
        )

        self.sigmoid = nn.Sigmoid()

        self.tensor_to_PIL = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage()]
        )

    def forward(self, x):
        out = self.sigmoid(self.model(x))
        return out

    def loss_functions(self, prediction: torch.tensor, target: torch.tensor):

        # mse_loss = self.mse_loss(prediction, target)
        # psnr_loss = self.psnr_loss(prediction, target)
        ssim_loss = self.ssim_loss(prediction, target)
        return {
            #'mse': mse_loss,
            #'psnr': psnr_loss,
            "ssim": ssim_loss
        }

    def _step(self, batch, log_mode: str = "train"):
        _, attribute, image = batch
        prediction = self(attribute)
        loss_dict = self.loss_functions(prediction, image)
        for key in loss_dict:
            self.log(
                f"{log_mode}_{key}_loss", loss_dict[key], on_step=False, on_epoch=True
            )
            if key == self.target_loss:
                loss = loss_dict[key]
                self.log(f"{log_mode}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, log_mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, log_mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, log_mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
