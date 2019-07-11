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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU(v)]
            else:
                layers += [conv2d, nn.PReLU(v)]
            
            in_channels = v

    return nn.Sequential(*layers)

def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg['VGG16_bn'], batch_norm=True), **kwargs)
    return model
    
def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg['VGG19_bn'], batch_norm=True), **kwargs)
    return model

