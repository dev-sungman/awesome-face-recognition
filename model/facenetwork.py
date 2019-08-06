import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model.vgg import vgg19
from model.resnet import ResNet50
from model.arcface import Arcface
from model.flatter import Flatter

class FaceNetwork(nn.Module):
    def __init__(self, device, backbone, head, class_num, embedding_size):
        super(FaceNetwork, self).__init__()
        
        self.device = device
        self.class_num = class_num

        # select backbone network 
        if backbone == 'vgg':
            self.backbone = vgg19().to(self.device)
        elif backbone == 'resnet':
            self.backbone = ResNet50().to(self.device)

        self.flatter = Flatter(embedding_size=embedding_size).to(self.device)
        
        # select head network
        if head == 'arcface':
            self.head = Arcface(num_classes = self.class_num).to(self.device)

    def train_model(self, img, label, optimizer):
        feature_map = self.backbone(img)
        embeddings = self.flatter(feature_map)
        thetas = self.head(embeddings, label)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(thetas, label)

        return loss
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.flatter(x)

        return x
