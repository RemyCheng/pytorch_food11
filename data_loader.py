from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'evaluation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = '/root/datasets/food11_dataset'

'''
return: dataloaders
{
    'train': [images, labels],
    'validation': [images, labels],
    'evaluation': [images, labels]
}
'''
def fetch_food11_dataloader(batch_size=64, num_workers=2):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'validation', 'evaluation']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['train', 'validation', 'evaluation']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'evaluation']}
    class_names = image_datasets['train'].classes
    return dataloaders



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

## usage
## for phase in ['train', 'validation']:
##      for inputs, labels in dataloaders[phase]:
##              inputs = inputs.to(device)
##              labels = labels.to(device)
##              ...

