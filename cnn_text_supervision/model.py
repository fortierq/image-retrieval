# Loads the ResNet50 model from MyResNet
# MyResnet is the default ResNet50 torchivision definition, but customize to be able to change its number of outputs and still load ImageNet weights
# Weights pretrained on ImageNet are used. Else the training takes ages.
# Model_Test is the same but includes the Sigmoid function to outputs, which during training is performed by the Cross Entropy Loss used (BCEWithLogitsLoss)

import torch.nn as nn
import MyResNet
import torch
import torch.nn.functional as F
from collections import OrderedDict


class Model(nn.Module):

    def __init__(self, embedding_dimensionality):
        super(Model, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=embedding_dimensionality)

    def forward(self, image):
        x = self.cnn(image)
        return x


class Model_Test(nn.Module):

    def __init__(self, embedding_dimensionality):
        super(Model_Test, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=False, num_classes=embedding_dimensionality)

    def forward(self, image):
        x = self.cnn(image)
        x = F.sigmoid(x) # During training the Sigmoid operator is included in the Cross-Entropy Loss
        return x