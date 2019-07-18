import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import sigmoid

import numpy as np
import math

class Denseface(nn.Module):
    def __init__(self, num_classes):
        super(Denseface, self).__init__()

        # U-Net branch
        self.unet = UNet(n_channels=3, n_classes=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.last_conv = nn.Conv2d(512, 512, 7)
        self.last_prelu = nn.PReLU(512)
        self.classifier = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)

        # Arcface branch
        self.head = Arcface(num_classes=num_classes)


    def train_model(self, img, mask, label, optimizer):
        #emb = self.vgg(img)
        emb = self.unet.feature_extraction(img) # 5, 512, 24, 24
        thetas = self.head(emb, label)

        cls_criterion = nn.CrossEntropyLoss()
        cls_loss = cls_criterion(thetas, label)

        predict_mask = self.unet(img)

        #mask_criterion = nn.CrossEntropyLoss()
        mask_criterion = nn.BCELoss()
        mask_loss = mask_criterion(predict_mask, mask)

        total_loss = cls_loss + mask_loss

        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()

        return total_loss.item()

    def forward(self, img): 
        x = self.unet.feature_extraction(img)
        x = self.avgpool(x)
        x = self.last_conv(x)
        x = self.last_prelu(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.bn(x)
        
        return x
