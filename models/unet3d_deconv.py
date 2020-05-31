# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

import sys

class UNet3D_ef_deconv(nn.Module):
    """
    Unet3d half transpose convolution network architecture
    """
    def __init__(self, in_channels=3, out_channels=1, init_features=30):
        super(UNet3D_ef_deconv, self).__init__()
        features = init_features
        self.bottleneck = UNet3D_ef_deconv._deconvblock(features * 8, features * 16, name="bottleneck")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="denc4")
        
    
    @staticmethod
    def _deconvblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features)),
                    (
                        name + "deconv1",
                        nn.ConvTranspose3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features)),
                    (
                        name + "deconv2",
                        nn.ConvTranspose3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    )
                ]
            )
        )
