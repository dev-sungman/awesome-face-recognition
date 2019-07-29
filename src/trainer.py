from src.data_handler import FaceLoader
from src.verification import evaluate
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
    def __init__(self, device, dataloader, backbone, head, log_dir, model_dir, embedding_size, batch_size):
        self.step = 0
        self.device = device
        
        self.embedding_size = embedding_size

        self.batch_size = batch_size

        self.train_loader, self.class_num = dataloader.get_loader()
        self.model = FaceNetwork(backbone, head, dataloader, embedding_size, init_weights=True)
        
        # For model saving
        self.model_dir = model_dir

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        self.writer = SummaryWriter(log_dir)
        
        self.board_loss_every = len(self.train_loader) // 1
        self.evaluate_every = len(self.train_loader) // 5
        self.save_every = len(self.train_loader) // 10
        
        print(self.board_loss_every, self.evaluate_every, self.save_every)

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

                if self.step % self.board_loss_every == 0 & self.step != 0:
                    self.writer.add_scalar('loss', loss, self.step)

                if self.step % self.evaluate_every == 0 & self.step != 0:
                    acc, best_thresh = self.evaluate(self.agedb_30, self.agedb_30_pair)
                    self.board_val('agedb_30', acc, best_thresh)

                    acc, best_thresh = self.evaluate(self.lfw, self.lfw_pair)
                    self.board_val('lfw', acc, best_thresh)
                    
                    acc, best_thresh = self.evaluate(self.cfp_fp, self.cfp_fp_pair)
                    self.board_val('cfp_fp', acc, best_thresh)

                if self.step % self.save_every == 0 & self.step != 0:
                    torch.save(self.model.state_dict(), self.model_dir + '/' + str(self.step) + '.pth')

                if self.step == 20000:
                    for params in self.optimizer.param_groups:
                        params['lr'] /= 10

                elif self.step == 28000:
                    for params in self.optimizer.param_groups:
                        params['lr'] /= 10

                self.step += 1


    def evaluate(self, carray, issame, nrof_folds=5):
        self.model.eval()
        embeddings = np.zeros([len(carray), 512])

        idx = 0
        with torch.no_grad():
            while idx + self.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx+self.batch_size])
                embeddings[idx:idx+self.batch_size] = self.model(batch.to(self.device)).cpu()

                idx += self.batch_size

            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                embeddings[idx:] = self.model(batch.to(self.device)).cpu()

        tpr, fpr, acc, best_thresh = evaluate(embeddings, issame, nrof_folds)

        self.model.train()

        return acc.mean(), best_thresh.mean()

    def board_val(self, db_name, acc, best_thresh):
        self.writer.add_scalar('{}_accuracy'.format(db_name), acc, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_thresh, self.step)

