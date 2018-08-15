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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fetch_teacher_outputs(dataloader, teacher_model_version):
    if teacher_model_version == 'resnet18':
        teacher_model = models.resnet18()
    teacher_model.to(device).eval()
    teacher_outputs = []	
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
		labels = labels.to(device)
	    outputs = teacher_model(inputs).detach().to("cpu").numpy()
        teacher_outputs.append(outputs)
    return teacher_outputs
def kdloss(outputs, labels, teacher_outputs, alpha=0.7, temperature=2):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
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
	return model, update_params
def set_model_and_optimization(params):
    require_teacher = False
	
	if params.model_version == 'resnet18':
        if params.pretrained == "yes":
            model = models.resnet18(pretrained=True)
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, 11)
            if params.freeze_conv == 'yes':
                model, update_params = freeze_resnet_conv(model)
			else:
                model = models.resnet18(num_classes=11)
				update_params = model.parameters()
        # optimizer
        optimizer = optim.SGD(update_params, lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
        # Set learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # Set loss function
        criterion = nn.CrossEntropyLoss()
    elif params.model_version == 'resnet8':
        if params.pretrained == "yes":
            model = resnet_cifar.resnet8(num_classes=11)
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, 11)
            if params.freeze_conv == 'yes':
                model, update_params = freeze_resnet_conv(model)
			else:
                model = resnet_cifar.resnet8(num_classes=11)
				update_params = model.parameters()
        if restore
        # optimizer
        optimizer = optim.SGD(update_params, lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
        # Set learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # Set loss function
        if params.teacher != "None":
            require_teacher = True
            criterion = lambda outputs, labels, teacher_outputs: kdloss(outputs, labels, teacher_outputs, params.alpha, params.temparature)
        else:		
            criterion = nn.CrossEntropyLoss()
    return model, scheduler, optimizer, criterion, require_teacher