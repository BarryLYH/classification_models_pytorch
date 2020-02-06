import torch
import torch.nn as nn

__all__ = ["resnet18", "resnet34", "resnet34", "resnet34", "resnet50"]

class Basicblock(nn.Module):
    expansion = 1#the change of depth after this block 
    #basicblock is the block for resnet18 and 34
    def __init__(self, dep_in, dep_out, stride = 1):
        super(Basicblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(dep_in, dep_out, 3, stride, 1),
            nn.BatchNorm2d(dep_out),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(dep_out, dep_out, 3, 1, 1),
            nn.BatchNorm2d(dep_out)
        )
        if stride != 1 or dep_out != dep_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dep_in, dep_out, 1, stride),
                nn.BatchNorm2d(dep_out)
            )
        else:
            self.shortcut = nn.Sequential()
        self.end_rule = nn.ReLU()

    def forward(self, x):
        fx = self.layer1(x)
        fx = self.layer2(fx)
        x = self.shortcut(x)
        x = x + fx
        x = self.end_rule(x)
        return x

class Bottleneckblock(nn.Module):
    expansion = 4
    # Bottleneckblock is the block for resnet50, 101 and 152 
    def __init__(self, dep_in, dep_mid, stride = 1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(dep_in, dep_mid, 1, stride),
            nn.BatchNorm2d(dep_mid),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(dep_mid, dep_mid, 3, 1, 1),
            nn.BatchNorm2d(dep_mid),
            nn.ReLU(inplace=True)   
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(dep_mid, dep_mid*4, 1, 1),
            nn.BatchNorm2d(dep_mid*4)
        )
        if stride != 1 or dep_in != dep_mid*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dep_in, dep_mid*4, 1, stride),
                nn.BatchNorm2d(dep_mid*4)
            )
        else:
            self.shortcut = nn.Sequential()
        self.end_rule = nn.ReLU()

    def forward(self, x):
        fx = self.layer1(x)
        fx = self.layer2(fx)
        fx = self.layer3(fx)
        x = self.shortcut(x)
        x = self.end_rule(x)
        return  x

class Resnet(nn.Module):
    def __init__(self, block, c2, c3, c4, c5, class_num = 1000):
        super(Resnet, self).__init__()
        # c2, c3, c4, c5 here mean the block number in conv2_x,
        # conv3_x, conv4_x, conv5_x
        
        #Here the input_size of image is 3*227*227. After first conv2, 
        #the size of features will turn to be resnet size.
        self.pre = nn.Sequential(
            #the original resnet input size is 3*224*224
            #nn.Conv2(3, 64, 7, 2, 3) 
            nn.Conv2d(3, 64, 7, 2, 1), # 64*112*112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace =True),
            nn.MaxPool2d(3, 2, 1) # 64*56*56
        )
        self.last_depth = 64
        self.conv2_x = self.combine_blocks(block, c2, 64,  1)
        self.conv3_x = self.combine_blocks(block, c3, 128, 2)
        self.conv4_x = self.combine_blocks(block, c4, 256, 2)
        self.conv5_x = self.combine_blocks(block, c5, 512, 2)

        self.out = nn.Sequential(
            nn.AvgPool2d(7, 1),
            nn.Dropout(p=0.4),
        )
        self.fc_layer = nn.Linear(block.expansion*512, class_num)
    
    def combine_blocks(self, block, block_num, depth, stride):
        strides = [stride] + [1] * (block_num -1)
        layers = []
        for s in strides:
            layers.append(block(self.last_depth, depth, s))
            self.last_depth = depth * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        print(x.size())
        x = self.conv2_x(x)
        print(x.size())
        x = self.conv3_x(x)
        print(x.size())
        x = self.conv4_x(x)
        print(x.size())
        x = self.conv5_x(x)
        print(x.size())
        x = self.out(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.fc_layer(x)
        print(x.size())
        return x

def resnet18(**kwargs):
    model = Resnet(Basicblock, 2, 2, 2, 2, **kwargs)
    return model

def resnet34(**kwargs):
    model = Resnet(Basicblock, 2, 4, 6, 3, **kwargs)
    return model

def resnet50(**kwargs):
    model = Resnet(Bottleneckblock, 3, 4, 6, 3, **kwargs)
    return model

def resnet101(**kwargs):
    model = Resnet(Bottleneckblock, 3, 4, 23, 3, **kwargs)
    return model

def resnet152(**kwargs):
    model = Resnet(Bottleneckblock, 3, 8, 36, 3, **kwargs)
    return model