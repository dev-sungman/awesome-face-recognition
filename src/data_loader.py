import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

import numpy as np
import cv2
import sys
import os

IMG_EXTENSIONS=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def make_transforms(img_size):
    train_data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    mask_data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        ])

    return train_data_transform, mask_data_transform
    

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def make_dataset(origin_dir, mask_dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    origin_dir = os.path.abspath(origin_dir)
    mask_dir = os.path.abspath(mask_dir)
    
    if not ((extensions is not None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    
    # TODO: Need to optimization 
    for target in sorted(class_to_idx.keys()):
        origin_d = os.path.join(origin_dir, target)
        mask_d = os.path.join(mask_dir, target)

        if not (os.path.isdir(origin_d) | os.path.isdir(mask_d)):
            continue
        for root, _, fnames in sorted(os.walk(origin_d)):
            # below codelines are not best. 
            mask_root = root.split('/')[:-2]
            mask_root = '/'.join(mask_root) + '/gt/' + root.split('/')[-1]
            for name in sorted(fnames):
                origin_path = os.path.join(root, fname)
                mask_path = mask_root + '/' + fname
                if (is_valid_file(origin_path) & is_valid_file(mask_path)):
                    item = (origin_path, mask_path, class_to_idx[target])
                    images.append(item)

    return images


# if you want to training with binary mask, it will be used
class FaceDataset(Dataset):
    def __init__(self, train_root, mask_root=None, train_transform=None, mask_transform=None, extensions=None, transform=None, is_valid_file=None):
        self.train_root = train_root
        self.mask_root = mask_root

        self.train_transform = train_transform
        self.mask_transform = mask_transform

        classes, class_to_idx = self._find_classes(self.train_root)
        samples = make_dataset(self.train_root, self.mask_root, class_to_idx, extensions, is_valid_file)
        
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.train_root + "\n"
                "Supported extensions are: " + ",".join(extensions)))

        self.classes = classes
        self.samples = samples
        
    def __len__(self):
        return len(os.listdir(self.train_root))

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

        classes.sort()
        class_to_idx = {classes[i]:i for i in range(len(classes))}

        return classes, class_to_idx

    def __getitem__(self, idx):
        origin_path, mask_path, target = self.samples[idx]

        image = Image.open(origin_path)
        mask = Image.open(mask_path)

        if self.train_transform:
            image = self.train_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        sample = (image, mask, target)
     
        return sample
        

class FaceLoader:
    def __init__(self, batch_size, datasets, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.loader = DataLoader(datasets, batch_size=self.batch_size, num_workers=4, pin_memory=True, shuffle=True)
        
        self.num_classes = len(datasets)
