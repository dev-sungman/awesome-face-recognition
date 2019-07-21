from src.vgg import vgg19_bn
from src.arcface import Arcface
from src.data_loader import FaceLoader
from src.utils import get_val_pair, get_val_data

import torch
from torch import optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from PIL import Image
import math
import numpy as np
import os

from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm

class FaceTrainer:
    def __init__(self, device, dataloader, log_dir, model_dir, embedding_size=512):
        self.step = 0
        self.device = device
        
        self.embedding_size = embedding_size

        self.train_loader, self.class_num = dataloader.get_loader()
        self.model_dir = model_dir

        self.model = vgg19_bn(num_classes=self.class_num).to(self.device)
        self.model = nn.DataParallel(self.model)
        self.head = Arcface(num_classes = self.class_num).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    def train(self, epcohs):
        self.model.train()

        running_loss = 0.

        for epoch in range(epochs):
            print('epoch', epoch, ' started')
            for imgs, masks, labels in tqdm(iter(self.train_loader)):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)

                lossfunc = CrossEntropyLoss()
                loss = lossfunc(thetas, labels)
                loss.backward()

                self.optimizer.step()

                self.step += 1



