from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []
        for i in range(self.num_down_layers):
            #conv list
            in_channels = 3 if i == 0 else down_filter_sizes[i - 1]
            out_channels = down_filter_sizes[i]
            layers_conv_down.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              padding=conv_paddings[i], kernel_size=kernel_sizes[i]))

            #bn list
            layers_bn_down.append(nn.BatchNorm2d(num_features=out_channels))

            #MaxPool2d list
            layers_pooling.append(nn.MaxPool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i], return_indices=True))
        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []
        output_size = down_filter_sizes[-1]
        for i in range(self.num_up_layers):
            #MaxUnPool2d list
            layers_unpooling.append(nn.MaxUnpool2d(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i]))

            #conv list
            in_channels = output_size if i == 0 else up_filter_sizes[i - 1]
            out_channels = up_filter_sizes[i]
            layers_conv_up.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            padding=conv_paddings[i], kernel_size=kernel_sizes[i]))

            #bn list
            layers_bn_up.append(nn.BatchNorm2d(num_features=out_channels))

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to get the logits of 11 classes (background + 10 digits)
        self.conv = nn.Conv2d(in_channels=up_filter_sizes[-1], out_channels=11, padding=2, kernel_size=3)

    def forward(self, x):
        indices_list = []
        for i in range(self.num_down_layers):
            conv_down = self.layers_conv_down[i]
            bn_down = self.layers_bn_down[i]
            pooling = self.layers_pooling[i]
            x, indices = pooling(self.relu(bn_down(conv_down(x))))
            indices_list.append(indices)
        indices_list.reverse()
        for i in range(self.num_up_layers):
            unpooling = self.layers_unpooling[i]
            conv_up = self.layers_conv_up[i]
            bn_up = self.layers_bn_up[i]
            x = self.relu(bn_up(conv_up(unpooling(x, indices_list[i]))))
        x = self.conv(x)
        return x

def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
