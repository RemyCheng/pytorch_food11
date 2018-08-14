from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

import argparse
import logging
import random

import utils
import data_loader
import model.resnet as resnet
import model.resnet_cifar as resnet_cifar
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/resnet18/config1',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'

def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=100):
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
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += inputs.size(0)
                
        epoch_loss = running_loss / running_total
        epoch_acc = running_corrects.double() / running_total

        logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # validate
        val_acc = evaluate(dataloaders['validation'], model)
        logging.info('Acc: {:.4f}'.format(val_acc))		
        # deep copy the model
        if val_acc > best_acc:
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
    dataloaders = data_loader.fetch_food11_dataloader(batch_size=params.batch_size, num_workers=0)
    logging.info("- done.")
    
    '''
    Set Model and Optimization
    '''
    if params.model_version == "resnet18":
        model = resnet.ResNet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
        # Set learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        
        # Set loss function
        criterion = nn.CrossEntropyLoss()
    elif params.model_version == "resnet8":
        model = resnet_cifar.ResNet8().to(device)
        if args.restore_file is not None:
	        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
	        logging.info("Restoring parameters from {}".format(restore_path))
	        ck = utils.load_checkpoint(restore_path, model, None)
        if params.freeze_conv == 1:
            for param in model.parameters():
                param.requires_grad = False
            model.linear.weight.requires_grad = True
            model.linear.bias.requires_grad = True
            optimizer = optim.SGD(model.linear.parameters(), lr=params.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
        # Set learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # Set loss function
        criterion = nn.CrossEntropyLoss()
        

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    
   
    #train_and_evaluate(model, train_dl, dev_dl, scheduler, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)
    #model, best_acc = train_model(datasets, model, criterion, optimizer, scheduler, num_epochs=params.num_epochs, teacher_outputs=teacher_outputs)
    model, best_acc = train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=params.num_epochs)
    utils.save_checkpoint({'epoch': params.num_epochs + 1,
                           'state_dict': model.state_dict(),
                           'optim_dict' : optimizer.state_dict()},
                           is_best=best_acc,
                           checkpoint=args.model_dir)
