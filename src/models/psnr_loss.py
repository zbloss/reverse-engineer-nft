from piqa import PSNR


class PSNRLoss(PSNR):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)
