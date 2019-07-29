from model.vgg import vgg19
from model.arcface import Arcface
from model.flatter import Flatter
from src.data_handler import FaceLoader

import torch
from torch import optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from PIL import Image
import math
import numpy as np
import os

from tensorboardX import SummaryWriter

class FaceTrainer:
    def __init__(self, device, dataloader, backbone, head, log_dir, model_dir, embedding_size=512):
        self.step = 0
        self.device = device
        
        self.embedding_size = embedding_size

        self.train_loader, self.class_num = dataloader.get_loader()
        self.model_dir = model_dir
        
        if backbone == 'vgg':
            self.backbone = vgg19().to(self.device)
            self.backbone = nn.DataParallel(self.backbone)

        self.flatter = Flatter(embedding_size=embedding_size).to(self.device)
        
        self.head = Arcface(num_classes = self.class_num).to(self.device)

        self.optimizer = optim.SGD(self.backbone.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    def train(self, epochs):
        self.backbone.train()

        running_loss = 0.

        for epoch in range(epochs):
            print('epoch', epoch, ' started')
            for imgs, labels in iter(self.train_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                feature_map = self.backbone(imgs)
                embeddings = self.flatter(feature_map)
                thetas = self.head(embeddings, labels)

                lossfunc = CrossEntropyLoss()
                loss = lossfunc(thetas, labels)
                loss.backward()
                
                self.optimizer.step()

                self.step += 1



