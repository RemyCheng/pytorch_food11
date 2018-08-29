from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import model.resnet_cifar as resnet_cifar

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fetch_teacher_outputs(dataloader, teacher_model_version, ckpt_path):
    if teacher_model_version == 'resnet18':
        teacher_model = models.resnet18(num_classes=11)
    
    utils.load_checkpoint(checkpoint=ckpt_path, model=teacher_model)
    
    teacher_model.to(device).eval()
    teacher_outputs = []	
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = teacher_model(inputs).detach().to(torch.device('cpu')).numpy()
        teacher_outputs.append(outputs)
        #print(outputs.shape)
    return teacher_outputs
def fetch_teacher_model(teacher_model_version, ckpt_path):
    if teacher_model_version == 'resnet18':
        teacher_model = models.resnet18(num_classes=11)
    elif teacher_model_version == 'resnet34':
        teacher_model = models.resnet34(num_classes=11)
    elif teacher_model_version == 'resnet50':
        teacher_model = models.resnet50(num_classes=11)
    elif teacher_model_version == 'resnet152':
        teacher_model = models.resnet152(num_classes=11)
    utils.load_checkpoint(ckpt_path, teacher_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model
def kdloss(outputs, labels, teacher_outputs, alpha=0.7, temperature=2):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss        
def freeze_resnet_conv(model):
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    update_params = model.fc.parameters()
    return update_params
def configure_resnet_model(model, params):
    require_teacher = False
    # pre-trained from local pth file
    if params.pretrained != 'none' and params.pretrained != 'ImageNet':
        utils.load_checkpoint(params.pretrained, model)	
    # freeze conv layers
    if params.freeze_conv == 'yes':
        update_params = freeze_resnet_conv(model)
    else:
        update_params = model.parameters()
    # optimizer
    optimizer = optim.SGD(update_params, lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    # set learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params.lr_step, gamma=0.1)
    # set loss function
    if params.teacher != "none":
        require_teacher = True
        criterion = lambda outputs, labels, teacher_outputs: kdloss(outputs, labels, teacher_outputs, params.alpha, params.temperature)	
    else:
        criterion = nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion, require_teacher
def fetch_model_and_optimization(params):
    # resnet18
    if params.model_version == 'resnet18':
        # pretrained model of imagenet from pytorch model zoo
        if params.pretrained == "ImageNet":
            model = models.resnet18(pretrained=True)
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, 11)
        else:
            model = models.resnet18(num_classes=11)
        # configure freeze_conv, optimizer, scheduler, criterion, require_teacher			
        optimizer, scheduler, criterion, require_teacher = configure_resnet_model(model, params)
    # resnet34
    elif params.model_version == 'resnet34':
        # pretrained model of imagenet from pytorch model zoo
        if params.pretrained == 'ImageNet':
            model = models.resnet34(pretrained=True)
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, 11)
        # pretrained from local directory or no pretrained file
        else:
            model = models.resnet34(num_classes=11)
        # configure freeze_conv, optimizer, scheduler, criterion, require_teacher
        optimizer, scheduler, criterion, require_teacher = configure_resnet_model(model, params)
    # resnet50
    elif params.model_version == 'resnet50':
        # pretrained from ImageNet
        if params.pretrained == 'ImageNet':
            model = models.resnet50(pretrained=True)
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, 11)
        # pretrained file from local directory or no pretrained file
        else:
            model = models.resnet50(num_classes=11)
        # configure freeze_conv, optimizer, scheduler, criterion, require_teacher				
        optimizer, scheduler, criterion, require_teacher = configure_resnet_model(model, params)
    # resnet152
    elif params.model_version == 'resnet152':
        # pretrained from ImageNet
        if params.pretrained == 'ImageNet':
            model = models.resnet152(pretrained=True)
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, 11)
        # pretrained file from local directory or no pretrained file
        else:
            model = models.resnet152(num_classes=11)
        # configure freeze_conv, optimizer, scheduler, criterion, require_teacher 				
        optimizer, scheduler, criterion, require_teacher = configure_resnet_model(model, params)
    # resnet8
    elif params.model_version == 'resnet8':
        model = resnet_cifar.resnet8(num_classes=11)
        # configure freeze_conv, optimizer, scheduler, criterion, require_teacher 				
        optimizer, scheduler, criterion, require_teacher = configure_resnet_model(model, params)
    # resnet14
    elif params.model_version == 'resnet14':
        model = resnet_cifar.resnet14(num_classes=11)
        # configure freeze_conv, optimizer, scheduler, criterion, require_teacher 				
        optimizer, scheduler, criterion, require_teacher = configure_resnet_model(model, params)
    # resnet20
    elif params.model_version == 'resnet20':
        model = resnet_cifar.resnet20(num_classes=11)
        # configure freeze_conv, optimizer, scheduler, criterion, require_teacher 				
        optimizer, scheduler, criterion, require_teacher = configure_resnet_model(model, params)
    
    return model, scheduler, optimizer, criterion, require_teacher
