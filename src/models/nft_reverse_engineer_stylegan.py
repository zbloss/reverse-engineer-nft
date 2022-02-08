import torch
import torch.nn as nn
from .noise import GANNoise
from .aidn_layer import AdaINLayer
from .psnr_loss import PSNRLoss
from .ssim_loss import SSIMLoss


class NFTReverseEngineerStyleGAN(nn.Module):
    def __init__(self, target_loss: str = "ssim", use_cuda: bool = False):
        super().__init__()

        self.target_loss = target_loss
        self.use_cuda = use_cuda

        # 8-layer MLP similar to StyleGAN that creates
        # a normalized nonlinear latent vector with dim
        # 1024 that we can begin doing image generation
        # from.

        self.fully_connected = nn.Sequential(
            nn.Linear(21, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
        )

        # initial block
        self.constant = torch.rand((4, 1024, 4, 4))
        if self.use_cuda and torch.cuda.is_available():
            self.constant = self.constant.cuda()

        self.adain = AdaINLayer()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=(1, 1))

        # block 1
        self.upsample1 = nn.Upsample((8, 8), mode="bilinear")
        self.conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1))

        # block 2
        self.upsample2 = nn.Upsample((16, 16), mode="bilinear")
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))

        # block 3
        self.upsample3 = nn.Upsample((32, 32), mode="bilinear")
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))

        # block 4
        self.upsample4 = nn.Upsample((64, 64), mode="bilinear")
        self.conv8 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv9 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))

        # block 5
        self.upsample5 = nn.Upsample((128, 128), mode="bilinear")
        self.conv10 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv11 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1))

        # block 6
        self.upsample6 = nn.Upsample((256, 256), mode="bilinear")
        self.conv12 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv13 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=(1, 1))

        # block 7
        self.upsample7 = nn.Upsample((512, 512), mode="bilinear")
        self.conv14 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=(1, 1))
        self.conv15 = nn.Conv2d(8, 4, kernel_size=(3, 3), padding=(1, 1))

        # block 8
        self.upsample8 = nn.Upsample((1024, 1024), mode="bilinear")
        self.conv16 = nn.Conv2d(4, 4, kernel_size=(3, 3), padding=(1, 1))
        self.conv17 = nn.Conv2d(4, 3, kernel_size=(3, 3), padding=(1, 1))

        # final block
        self.upsample9 = nn.Upsample((1536, 1024), mode="bilinear")
        self.conv18 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))
        self.conv19 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))

        self.noise_layer = GANNoise()

        self.mse_loss = nn.MSELoss()
        self.psnr_loss = PSNRLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, x):

        x = x.unsqueeze(1).unsqueeze(1)

        # initial block
        self.constant += self.noise_layer(self.constant.shape)
        # self.constant += self.get_noise(self.constant.shape)
        latent_vector = self.fully_connected(x)
        x = self.adain(self.constant, latent_vector)
        x = self.conv1(x)
        x = self.adain(x, latent_vector)

        # block 1
        x = self.upsample1(x)
        x = self.conv2(x)
        # x += self.get_noise(x.shape)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv3(x)
        # x += self.get_noise(x.shape)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # block 2
        x = self.upsample2(x)
        x = self.conv4(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv5(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # block 3
        x = self.upsample3(x)
        x = self.conv6(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv7(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # block 4
        x = self.upsample4(x)
        x = self.conv8(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv9(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # block 5
        x = self.upsample5(x)
        x = self.conv10(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv11(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # block 6
        x = self.upsample6(x)
        x = self.conv12(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv13(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # block 7
        x = self.upsample7(x)
        x = self.conv14(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv15(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # block 8
        x = self.upsample8(x)
        x = self.conv16(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv17(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        # final block
        x = self.upsample9(x)
        x = self.conv18(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)
        x = self.conv19(x)
        x += self.noise_layer(x.shape)
        x = self.adain(x, latent_vector)

        return x

    def get_noise(self, size):
        noise = torch.rand(size)
        if torch.cuda.is_available():
            noise = noise.cuda()

        return noise

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
