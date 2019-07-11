import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

__all__ = ['VGG', 'vgg16_bn', 'vgg19_bn',]

cfg = {
        'VGG16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

class VGG(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.last_conv = nn.Conv2d(512, 512, 7)
        self.last_prelu = nn.PReLU(512)
        self.linear = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.last_prelu(self.last_conv(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        
        # l2 normalization
        x = torch.div(x, torch.norm(x, 2, axis=1, True))
