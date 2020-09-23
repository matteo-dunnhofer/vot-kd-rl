"""
Written by Matteo Dunnhofer - 2020

ResNet classes definitions
"""
import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):

    def __init__(self, cfg, training=False):
        super(ResNet18, self).__init__()

        self.name = 'ResNet18'
        self.is_training = training
        self.cfg = cfg

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2] # delete the last fc layer (and the avg pool for smaller input sizes)
        self.cnn_features = nn.Sequential(*modules)

        del resnet

    def forward(self, x):
        return self.cnn_features(x)

