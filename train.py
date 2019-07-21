import torch 
import torch.nn as nn

import sys
import os
import argparse

from src.utils import *

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    # set up root for training dataset
    parser.add_argument('--train_root', type=str, default=None)
    # set up root for training dataset (masked)
    parser.add_argument('--mask_root', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    # set up root for saving model, log
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--gpu_idx', type=int, default=0)

    return parser.parse_args(argv)

def main(args):
    
    # check cuda availablity
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # make save directory
    model_dir = args.save_root + '/model'
    make_dir(model_dir) 

    log_dir = args.save_root + '/log'
    make_dir(log_dir)
    
    # create dataset, dataloader
    face_datasets =  

    # train using trainer

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
