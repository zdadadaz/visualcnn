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
    def __init__(self):
        super(UNet3D_ef_deconv, self).__init__()
        features = 30
        in_channels=3
     
        fin = [features * 16, features * 8, features * 4, features * 2, features]
        fout = [features * 8, features * 4, features * 2, features, in_channels]
        self.features = nn.Sequential(
            # encoder 1
            nn.ConvTranspose3d(fin[0], fin[0], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[0],out_channels=fin[0],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose3d(fin[0], fout[0], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[0],out_channels=fout[1],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.ConvTranspose3d(fout[0], fout[0], kernel_size=2, stride=2),
            nn.MaxUnpool3d(2, stride=2),
            #encoder 2
            nn.ConvTranspose3d(fin[1], fin[1], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[1],out_channels=fin[1],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose3d(fin[1], fout[1], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[1],out_channels=fout[1],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.ConvTranspose3d(fout[1], fout[1], kernel_size=2, stride=2),
            nn.MaxUnpool3d(2, stride=2),
            #encoder 3
            nn.ConvTranspose3d(fin[2], fin[2], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[2],out_channels=fin[2],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose3d(fin[2], fout[2], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[2],out_channels=fout[2],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.ConvTranspose3d(fout[2], fout[2], kernel_size=2, stride=2),
            nn.MaxUnpool3d(2, stride=2),
            #encoder 4
            nn.ConvTranspose3d(fin[3], fin[3], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[3],out_channels=fin[3],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose3d(fin[3], fout[3], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[3],out_channels=fout[3],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.ConvTranspose3d(fout[3], fout[3], kernel_size=2, stride=2),
            nn.MaxUnpool3d(2, stride=2),
            #encoder 5
            nn.ConvTranspose3d(fin[4], fin[4], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[4],out_channels=fin[4],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose3d(fin[4], fout[4], 3, padding=1, bias = False),
            # nn.Conv3d(in_channels=fin[4],out_channels=fout[4],kernel_size=3,padding=1,stride=1,bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
        
        self.conv2deconv_indices = {
                0:22, 3:20, 7:17, 10:15, 14:12, 17:10, 21:7, 24:5, 28:2, 31:0
                }

        self.unpool2pool_indices = {
                6:19, 13:14, 20:9, 27:4
                }

        self.init_weight()
        # for idx, layer in enumerate(self.features):
        #     if isinstance(layer, nn.MaxUnpool3d):
        #         print(idx,layer)
    
    def init_weight(self):
        checkpoint = torch.load("/home/zdadadaz/Desktop/course/medical/code/echodyn/output/video/echonet_ef/best.pt")
        # model.load_state_dict(checkpoint['state_dict'])
        # Print model's state_dict
        # print("Model's state_dict:")
        count = 0
        idxlist = list(self.conv2deconv_indices.keys())
        for idx, layer in enumerate(checkpoint['state_dict']):
            if idx<10:
                self.features[self.conv2deconv_indices[idxlist[count]]].weight.data = checkpoint['state_dict'][layer]
                count += 1
            
    def forward(self, x, layer, activationin_channels_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool3d):
                x = self.features[idx](x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                print(idx)
                x = self.features[idx](x)
        return x
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=in_channels)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    # (name + "norm1", nn.BatchNorm2d(num_features=in_channels)),
                    # (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    # (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    # (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
# UNet3D_ef_deconv()