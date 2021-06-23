# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:25:18 2020

@author: Zhe
"""

from typing import Tuple, List

import torch
import torch.nn as nn
from jarvis.models.resnet import ResNet

class BlurNet(ResNet):
    r"""BlurNet model.

    A ResNet model with a fixed blurring preprocessing layer.

    """

    def __init__(self, sigma: float = 1.5, **kwargs) -> None:
        super(BlurNet, self).__init__(**kwargs)
        in_channels = self.in_channels
        self.sigma = sigma

        if sigma is None:
            self.blur = nn.Sequential()
        else:
            half_size = int(-(-2.5*sigma//1))
            x, y = torch.meshgrid(
                torch.arange(-half_size, half_size+1).to(torch.float),
                torch.arange(-half_size, half_size+1).to(torch.float)
                )
            w = torch.exp(-(x**2+y**2)/(2*sigma**2))
            w /= w.sum()
            self.blur = nn.Conv2d(
                in_channels, in_channels, 2*half_size+1,
                padding=half_size, padding_mode='circular', bias=False
                )
            self.blur.weight.data *= 0
            for i in range(in_channels):
                self.blur.weight.data[i, i] = w
            self.blur.weight.requires_grad = False

    def layer_activations(
            self,
            x: torch.Tensor
            ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        r"""Returns activations of all layers.

        Args
        ----
        x: (N, C, H, W), tensor
            The normalized input images.

        Returns
        -------
        pre_acts, post_acts: list of tensors
            The pre and post activations for each layer.
        logits: tensor
            The logits.

        """
        return super(BlurNet, self).layer_activations(self.blur(x))

def blurnet18(**kwargs):
    return BlurNet(block_nums=[2, 2, 2, 2], block_type='Basic', **kwargs)


def blurnet34(**kwargs):
    return BlurNet(block_nums=[3, 4, 6, 3], block_type='Basic', **kwargs)


def blurnet50(**kwargs):
    return BlurNet(block_nums=[3, 4, 6, 3], block_type='Bottleneck', **kwargs)


def blurnet101(**kwargs):
    return BlurNet(block_nums=[3, 4, 23, 3], block_type='Bottleneck', **kwargs)


def blurnet152(**kwargs):
    return BlurNet(block_nums=[3, 8, 36, 3], block_type='Bottleneck', **kwargs)
