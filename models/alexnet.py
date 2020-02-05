import torch
import torch.nn as nn
import numpy as np

__all__ = ['alexnet']

class Alexnet(nn.Module):
    def __init__(self, class_num = 1000):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            #input size 3*227*227
            #in_channels, out_channels, kernal_size, stride, padding
            nn.Conv2d(3, 96, 11, 4, 0), #55*55*96
            nn.ReLU(inplace=True),
            #kernel_size, stride
            nn.MaxPool2d(3, 2), #27*27*96
            nn.Conv2d(96, 256, 5, 1, 2), #27*27*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), #13*13*256
            nn.Conv2d(256, 384, 3, 1, 1), #13*13*384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1), #13*13*384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1), #13*13*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), #6*6*256
        )
        #in_features=6*6*256, out_features=4096
        self.classifier = nn.Sequential(
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, class_num)
        )

    def forward(self, x):
        x = self.features(x)
        #fatten x in two size [batch_size, -1]
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

def alexnet(**kwargs):
    model = Alexnet(**kwargs)
    return model