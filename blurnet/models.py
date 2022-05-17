from typing import Optional

import torch
import torch.nn as nn
from jarvis.models import ImageClassifier, ResNet

Tensor = torch.Tensor

class BlurNet(ImageClassifier):
    r"""BlurNet model.

    A ResNet model with a fixed blurring preprocessing layer.

    """

    def __init__(self,
        resnet: ResNet,
        sigma: Optional[float] = 1.5,
    ):
        r"""
        Args
        ----
        resnet:
            A ResNet model.
        sigma:
            Gaussian blurring parameter.

        """
        super(BlurNet, self).__init__()
        self.resnet = resnet
        self.in_channels = self.resnet.in_channels
        self.num_classes = self.resnet.num_classes
        self.normalizer = self.resnet.normalizer

        self.sigma = sigma
        if sigma is None:
            self.blur = nn.Sequential()
        else:
            half_size = int(-(-2.5*sigma//1))
            x, y = torch.meshgrid(
                torch.arange(-half_size, half_size+1).to(torch.float),
                torch.arange(-half_size, half_size+1).to(torch.float),
                indexing='ij',
                )
            w = torch.exp(-(x**2+y**2)/(2*sigma**2))
            w /= w.sum()
            self.blur = nn.Conv2d(
                self.in_channels, self.in_channels, 2*half_size+1,
                padding=half_size, padding_mode='circular', bias=False,
            )
            self.blur.weight.data *= 0
            for i in range(self.in_channels):
                self.blur.weight.data[i, i] = w
            self.blur.weight.requires_grad = False

    def layer_activations(self, x: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
        return self.resnet.layer_activations(self.blur(x))
