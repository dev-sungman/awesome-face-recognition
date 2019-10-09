import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from model.non_local_embedded_gaussian import NONLocalBlock2D

__all__ = ['VGG', 'vgg16', 'vgg19',]

cfg = {
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

class VGG(nn.Module):
    def __init__(self, features, init_weights=True, embedding_size=512):
        super(VGG, self).__init__()
        self.features = features
        self.non_local = NONLocalBlock2D(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, embedding_size)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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

def vgg16(**kwargs):
    model = VGG(make_layers(cfg['VGG16'], batch_norm=True), **kwargs)
    return model
    
def vgg19(**kwargs):
    model = VGG(make_layers(cfg['VGG19'], batch_norm=True), **kwargs)
    return model

def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1)

