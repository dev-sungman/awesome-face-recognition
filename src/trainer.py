from src.data_handler import FaceLoader
from model.facenetwork import FaceNetwork

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
    def __init__(self, device, dataloader, backbone, head, log_dir, model_dir, embedding_size):
        self.step = 0
        self.device = device
        
        self.embedding_size = embedding_size

        self.train_loader, self.class_num = dataloader.get_loader()
        self.model = FaceNetwork(backbone, head, dataloader, embedding_size, init_weights=True)
        
        # For model saving
        self.model_dir = model_dir

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    def train(self, epochs):

        running_loss = 0.

        for epoch in range(epochs):
            for imgs, labels in iter(self.train_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                
                loss = self.model.train_model(img=imgs, label=labels, optimizer=self.optimizer)
                

                self.optimizer.step()
                
                print('epoch', epoch, ' started', ' loss: ', loss)

                self.step += 1


    def eval(self)
