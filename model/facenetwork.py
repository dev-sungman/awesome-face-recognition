import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vgg import vgg19
from model.arcface import Arcface
from model.flatter import Flatter

class FaceNetwork(nn.Module):
    def __init__(self, backbone, head, data_loader, embedding_size, init_weights=True):
        # select backbone network 
        if backbone ='vgg':
            self.backbone = vgg19_bn(num_classes=self.class_num).to(self.device)
            self.backbone = nn.DataParallel(self.backbone)

        self.flatter = Flatter(embedding_size=embedding_size)
        
        # select head network
        if head = 'arcface':
            self.head = Arcface(num_classes = self.class_num).to(self.device)

        if init_weights:
            self._initialize_weights()
    
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

    def train_model(self, img, label, optimizer):
        feature_map = self.backbone(imgs)
        embeddings = self.flatter(feature_map)
        
        with torch.no_grad():
            thetas = self.head(embeddings, labels)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(thetas, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        return loss.item()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.flatter(x)

        return x
