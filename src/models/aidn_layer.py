import torch
import torch.nn as nn


class AdaINLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def height_weight_mean(self, x):
        """Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""

        numerator = torch.sum(x, (2, 3))
        denominator = x.shape[2] * x.shape[3]

        return numerator / denominator

    def height_weight_std(self, x, const: float = 2.3e-8):
        """Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""

        x_perm_minus_mean = x.permute([2, 3, 0, 1]) - self.height_weight_mean(x)
        x_perm_minus_mean = x_perm_minus_mean.permute([2, 3, 0, 1])
        x_perm_minus_mean = x_perm_minus_mean**2

        numerator = torch.sum(x_perm_minus_mean, (2, 3)) + const
        denominator = x.shape[2] * x.shape[3]

        return torch.sqrt(numerator / denominator)

    def forward(self, content_embedding, style_embedding):
        """Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""

        style_embedding_std = self.height_weight_std(style_embedding)
        style_embedding_mean = self.height_weight_mean(style_embedding)

        content_embedding_std = self.height_weight_std(content_embedding)
        content_embedding_mean = (
            content_embedding.permute([2, 3, 0, 1])
            - self.height_weight_mean(content_embedding)
        ) / content_embedding_std

        style_content_embedding = (
            style_embedding_std * content_embedding_mean + style_embedding_mean
        )

        return style_content_embedding.permute([2, 3, 0, 1])
