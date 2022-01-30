import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from piqa import PSNR, SSIM
from piqa.utils import set_debug
set_debug(False)

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

class PSNRLoss(PSNR):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

class NFTReverseEngineer(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, target_loss: str = 'psnr'):
        super().__init__()
        self.learning_rate = learning_rate
        self.target_loss = target_loss
        
        self.mse_loss = nn.MSELoss()
        self.psnr_loss= PSNRLoss()
        self.ssim_loss = SSIMLoss()
        
        self.width_conv1 = nn.Conv1d(21, 512, kernel_size=1)
        self.width_conv2 = nn.Conv1d(512, 1024, kernel_size=1)
        self.width_batch_norm1 = nn.BatchNorm1d(512)
        self.width_batch_norm2 = nn.BatchNorm1d(1024)
        
        self.height_conv1 = nn.Conv1d(1, 256, kernel_size=1)
        self.height_conv2 = nn.Conv1d(256, 512, kernel_size=1)
        self.height_conv3 = nn.Conv1d(512, 1536, kernel_size=1)
        self.height_batch_norm1 = nn.BatchNorm1d(256)
        self.height_batch_norm2 = nn.BatchNorm1d(512)
        self.height_batch_norm3 = nn.BatchNorm1d(1536)
        
        self.channel_conv = nn.Conv2d(1, 3, kernel_size=1)
        self.channel_batch_norm = nn.BatchNorm2d(3)

        self.linear = nn.Linear(1024, 1024)
        self.sigmoid = nn.Sigmoid()

        self.tensor_to_PIL = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage()
        ])
    
    def forward(self, x):
        
        x = x.unsqueeze(-1)
        x = self.width_conv1(x)
        x = self.width_batch_norm1(x)
        x = self.width_conv2(x)
        x = self.width_batch_norm2(x)

        x = x.view(-1, 1, x.shape[1])
        x = self.height_conv1(x)
        x = self.height_batch_norm1(x)
        x = self.height_conv2(x)
        x = self.height_batch_norm2(x)
        x = self.height_conv3(x)
        x = self.height_batch_norm3(x)
        
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        x = self.channel_conv(x)
        x = self.channel_batch_norm(x)
        
        x = self.linear(x)
        x = self.sigmoid(x)
        
        return x

    def loss_functions(self, prediction: torch.tensor, target: torch.tensor):

        #mse_loss = self.mse_loss(prediction, target)
        psnr_loss = self.psnr_loss(prediction, target)
        #ssim_loss = self.ssim_loss(prediction, target)
        return {
            #'mse': mse_loss,
            'psnr': psnr_loss,
            #'ssim': ssim_loss
        }
    
    def _step(self, batch, log_mode: str = 'train'):
        token_id, attribute, image = batch
        prediction = self(attribute)
        loss_dict = self.loss_functions(prediction, image)
        for key in loss_dict:
            self.log(f'{log_mode}_{key}_loss', loss_dict[key], on_step=False, on_epoch=True)
            if key == self.target_loss:
                loss = loss_dict[key]
                self.log(f'{log_mode}_loss', loss, on_step=True, on_epoch=True)
                
        return {'loss': loss, 'prediction': prediction}
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, log_mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, log_mode='val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, log_mode='test')

    def validation_epoch_end(self, outputs):
        
        save_path = os.path.join('./data/output', str(self.global_step))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        predictions = outputs[0]['prediction']
        for sample in predictions:
            prediction = sample
            image = self.tensor_to_PIL(prediction)
            artifact_path = os.path.join(save_path, f'{self.global_step}.png')
            image.save(artifact_path)
            self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
        shutil.rmtree(save_path)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate
        )
        return optimizer