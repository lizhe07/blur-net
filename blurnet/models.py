# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:25:18 2020

@author: Zhe
"""

import torch
import torch.nn as nn
from jarvis.models.resnet import ResNet

class BlurNet(ResNet):
    r"""BlurNet model.

    A ResNet model with a fixed blurring preprocessing layer.

    """

    def __init__(self, in_channels=3, blur_sigma=1.5, **kwargs):
        super(BlurNet, self).__init__(in_channels=in_channels, **kwargs)
        self.blur_sigma = blur_sigma

        if blur_sigma is not None:
            half_size = int(-(-2.5*blur_sigma//1))
            x, y = torch.meshgrid(
                torch.arange(-half_size, half_size+1).to(torch.float),
                torch.arange(-half_size, half_size+1).to(torch.float)
                )
            w = torch.exp(-(x**2+y**2)/(2*blur_sigma**2))
            w /= w.sum()
            blur = nn.Conv2d(in_channels, in_channels, 2*half_size+1,
                             padding=half_size, padding_mode='zeros', bias=False)
            blur.weight.data *= 0
            for i in range(in_channels):
                blur.weight.data[i, i] = w
            blur.weight.requires_grad = False
            self.sections[0] = nn.Sequential(
                blur, *self.sections[0]
                )

def blurnet18(**kwargs):
    return BlurNet([2, 2, 2, 2], 'Basic', **kwargs)


def blurnet34(**kwargs):
    return BlurNet([3, 4, 6, 3], 'Basic', **kwargs)


def blurnet50(**kwargs):
    return BlurNet([3, 4, 6, 3], 'Bottleneck', **kwargs)


def blurnet101(**kwargs):
    return BlurNet([3, 4, 23, 3], 'Bottleneck', **kwargs)


def blurnet152(**kwargs):
    return BlurNet([3, 8, 36, 3], 'Bottleneck', **kwargs)
