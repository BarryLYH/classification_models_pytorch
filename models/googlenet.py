import torch
import torch.nn as nn


__all__ = ["googlenet"]

class Inception(nn.Module):
    def __init__(self, input_depth, out1_1, mid3_3, out3_3,
                mid5_5, out5_5, max1_1):
        super(Inception, self).__init__()
        
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(input_depth, out1_1, 1, 1),
            nn.BatchNorm2d(out1_1),
            nn.ReLU(inplace=True)
        )
        self.layer3_3 = nn.Sequential(
            nn.Conv2d(input_depth, mid3_3, 1, 1),
            nn.BatchNorm2d(mid3_3),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid3_3, out3_3, 3, 1, 1),
            nn.BatchNorm2d(out3_3),
            nn.ReLU(inplace=True)
        )
        self.layer5_5 = nn.Sequential(
            nn.Conv2d(input_depth, mid5_5, 1, 1),
            nn.BatchNorm2d(mid5_5),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid5_5, out5_5, 5, 1, 2),
            nn.BatchNorm2d(out5_5),
            nn.ReLU(inplace=True)
        )
        self.layer3_1 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(input_depth, max1_1, 1, 1),
            nn.BatchNorm2d(max1_1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): 
        x1_1 = self.layer1_1(x)
        x3_3 = self.layer3_3(x)
        x5_5 = self.layer5_5(x)
        x3_1 = self.layer3_1(x)
        output = torch.cat((x1_1, x3_3, x5_5, x3_1), 1)
        return output


class GoogleNet(nn.Module):
    def __init__(self, class_num = 1000):
        super(GoogleNet, self).__init__()
        #Here the input_size of image is 3*227*227. After first conv2, 
        #the size of features will turn to be default googlenet size.
        self.pre = nn.Sequential(
            #the original googlenet input size is 3*224*224
            #nn.Conv2(3, 64, 7, 2, 3) 
            nn.Conv2d(3, 64, 7, 2, 1), # 64*112*112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace =True),
            nn.MaxPool2d(3, 2, 1), # 64*56*56
            nn.Conv2d(64, 192, 3, 1, 1), # 192*56*56
            nn.BatchNorm2d(192),
            nn.ReLU(inplace =True),
            nn.MaxPool2d(3, 2, 1) # 192*28*28
        )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)#256*28*28
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)#480*28*28

        self.maxpooling = nn.MaxPool2d(3, 2, 1)#480*14*14

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)#512*14*14
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)#512*14*14
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)#512*14*14
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)#528*14*14
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)#832*14*14
        #Here will be one more maxpooling layer
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)#832*7*7
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)#1024*7*7

        self.endlayer = nn.Sequential(
            nn.AvgPool2d(7, 1),#1024*1*1
            nn.Dropout(p=0.4),
        )
        self.fc_layer = nn.Linear(1024, class_num)

    def forward(self, x):
        x = self.pre(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpooling(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpooling(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.endlayer(x)
        x = x.view(x.size(0), 1024)
        x = self.fc_layer(x)
        return x


def googlenet(**kwargs):
    model = GoogleNet(**kwargs)
    return model
