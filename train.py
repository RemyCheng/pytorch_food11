from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import time
import os
import copy

import argparse
import logging
import random
from tqdm import tqdm

import utils
import data_loader, model_handler
import model.resnet as resnet
import model.resnet_cifar as resnet_cifar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/resnet18/test',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'

def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=100, teacher_model=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        # training phase
        scheduler.step()
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
            
        # Iterate over data.
        # tqdm progress bar
        with tqdm(total=len(dataloaders['train'])) as t:
            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if teacher_model == None:
                        loss = criterion(outputs, labels)
                    else:
                        teacher_model.eval()
                        teacher_model = teacher_model.to(device)
                        soft_targets = teacher_model(inputs)
                        #print(soft_targets[0])
                        #print(labels[0])
                        loss = criterion(outputs, labels, soft_targets)
                    loss.backward()
                    optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += inputs.size(0)
                t.update()
                
            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects.double() / running_total

            logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # validate
        val_acc = evaluate(dataloaders['validation'], model)
        logging.info('Val Acc: {:.4f}'.format(val_acc))		
        # deep copy the model
        if val_acc > best_acc:
            logging.info('Found new best accuracy')
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc
def evaluate(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)			
    acc = correct/total		
    return acc
if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    if torch.cuda.is_available(): torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch Food11 dataloaders {train, validation, evaluate}
    dataloaders = data_loader.fetch_food11_dataloader(batch_size=params.batch_size, num_workers=4)
    logging.info("- done.")
    
    '''
    Set Model and Optimization
    '''
    model, scheduler, optimizer, criterion, require_teacher = model_handler.fetch_model_and_optimization(params)
    model = model.to(device)
    if require_teacher:
        teacher_model = model_handler.fetch_teacher_model(params.teacher, params.teacher_ckpt_path)
    else:
        teacher_model = None
    
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    
    model, best_acc = train_model(dataloaders, model, criterion, optimizer, scheduler,
                                  num_epochs=params.num_epochs, teacher_model=teacher_model)
    utils.save_checkpoint({'epoch': params.num_epochs + 1,
                          'state_dict': model.state_dict()},
                          #'optim_dict' : optimizer.state_dict()},
                          is_best=best_acc,
                          checkpoint=args.model_dir)
    test_acc = evaluate(dataloaders['evaluation'], model)
    logging.info("Test Acc: {:.4f}".format(test_acc))
