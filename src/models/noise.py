import torch
import torch.nn as nn


class GANNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, size: tuple):
        noise = torch.rand(size)
        if torch.cuda.is_available():
            noise = noise.cuda()

        return noise
