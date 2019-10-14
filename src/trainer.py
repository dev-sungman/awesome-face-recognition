from src.utils import *
from src.verification import *
from torch.optim import lr_scheduler

from model.resnet import resnet18, resnet50
from model.arcface import ArcMarginProduct
from model.vgg import vgg19

import torch
from torch import optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import cv2
from PIL import Image
import math
import numpy as np
import os
import torchvision

from tensorboardX import SummaryWriter
from pathlib import Path
from torchsummary import summary

class FaceTrainer:
    def __init__(self, device, dataloader, backbone, head, log_dir, model_dir, batch_size, embedding_size=512):
        self.step = 0
        self.device = device
        self.batch_size = batch_size        
        self.embedding_size = embedding_size

        self.train_loader, self.class_num = dataloader.get_loader()
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        print('class number: ', self.class_num)
        
        if backbone == 'vgg':
            self.backbone = vgg19().to(self.device)
            self.margin = 10

        elif backbone == 'resnet':
            self.backbone = resnet50().to(self.device)
            self.margin = 15
        
        self.head = ArcMarginProduct(embedding_size, self.class_num, self.margin).to(self.device)
        print('backbone: ', backbone)
        
        self.optimizer = optim.SGD([
            {'params' : self.backbone.parameters()},
            {'params' : self.head.parameters()}], weight_decay=5e-4
            , lr=0.1, momentum=0.9)

        
        #self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_pair, self.cfp_fp_pair, self.lfw_pair = get_val_data(Path('data/eval/'))
        self.writer = SummaryWriter(log_dir)

        self.print_preq = 100
        self.board_loss_every = len(self.train_loader) // 5
        self.evaluate_every = len(self.train_loader) // 5
        self.save_every = len(self.train_loader) // 1

    def train(self, epochs):
        self.backbone.train()
        #self.exp_lr_scheduler.step()

        running_loss = 0.
        for epoch in range(epochs):
            print_step = 0
            for imgs, labels in iter(self.train_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.backbone(imgs)
                thetas = self.head(embeddings, labels)

                criterion = nn.CrossEntropyLoss()
                loss = criterion(thetas, labels)
                
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                
                # Save the training log
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step == 5000:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 10
                
                if self.step == 15000:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 10

                if self.step == 25000:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 10
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    
                    self.backbone.eval()
                    
                    acc, best_thresh = self.evaluate(self.agedb_30, self.agedb_30_pair, self.embedding_size)
                    self.board_val('agedb_30', acc, best_thresh)
                    #print("[AgeDB-30] acc: %0.4f\t best_thresh: %0.4f" %(acc, best_thresh))
                    
                    acc, best_thresh = self.evaluate(self.lfw, self.lfw_pair, self.embedding_size)
                    self.board_val('lfw', acc, best_thresh)
                    #print("[LFW] acc: %0.4f\t best_thresh: %0.4f" %(acc, best_thresh))
                    
                    acc, best_thresh = self.evaluate(self.cfp_fp, self.cfp_fp_pair, self.embedding_size)
                    self.board_val('cfp_fp', acc, best_thresh)
                    #print("[CFP-FP] acc: %0.4f\t best_thresh: %0.4f" %(acc, best_thresh))
                    
                    self.backbone.train()
                

                if self.step % self.print_preq == 0 and self.step != 0:
                    predicts = np.argmax(thetas.data.cpu().numpy(), axis=1)
                    gt = labels.data.cpu().numpy()

                    acc = np.mean((predicts == gt).astype(int))
                    print("[Epoch: %d\tIter: [%d/%d]\tLoss: %0.4f\t Acc: %0.2f]" %(epoch, print_step, len(self.train_loader), loss.item(), acc))

                self.step += 1
                print_step += 1
                    
                
            if epoch % 10 == 0 and epoch != 0:
                torch.save(self.backbone.state_dict(), self.model_dir + '/' + str(self.step) + '.pth')

    def evaluate(self, carray, issame, embedding_size, nrof_folds=5, tta=False):
        embeddings = np.zeros([len(carray), embedding_size])

        idx = 0
        with torch.no_grad():
            while idx + self.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx+self.batch_size])
                embeddings[idx:idx+self.batch_size] = self.backbone(batch.to(self.device)).cpu()
                
                idx += self.batch_size

            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                embeddings[idx:] = self.backbone(batch.to(self.device)).cpu()

        tpr, fpr, acc, best_thresh = evaluate(embeddings, issame, nrof_folds)
        
        return acc.mean(), best_thresh.mean()

    def board_val(self, db_name, acc, best_thresh):
        self.writer.add_scalar('{}_accuracy'.format(db_name), acc, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_thresh, self.step)

