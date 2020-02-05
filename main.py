import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.autograd import Variable
from models.alexnet import *
from models.googlenet import *
from utils.dataset import *


def main():
    epoch = 2
    chosen_model = "googlenet"
    transform = googlenet_transform()

    if chosen_model == "alexnet":
        model = alexnet(class_num = 2)
    elif chosen_model == "googlenet":
        model = googlenet(class_num = 2)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        print("Use GPU")
    else:
        print("Use CPU")
    train_txt_path = "/Users/Barry/Desktop/classification_models_pytorch/trainset.txt"
    trainset = Mydataset(train_txt_path, transform)
    trainloader = DataLoader(trainset, 
                             batch_size = 3, 
                             shuffle=True
                             )
    val_txt_path = "/Users/Barry/Desktop/classification_models_pytorch/valset.txt"
    valset = Mydataset(val_txt_path, transform)
    valloader = DataLoader(valset, 
                             batch_size = 3, 
                             shuffle=True
                             )
    
    print("data loaded")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001 )

    print("start to train")
    for i in range(epoch):
        model.train()
        for batch_index, (data, target) in enumerate(trainloader):
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            data, label = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_index > 0 and batch_index % 50 == 0:
                print("Trian epoch {},  iterate {}, loss is {}".format(i, batch_index, loss))
            # validate prcess
            if batch_index > 0 and batch_index % 500 == 0:
                model.eval()
                losses = []
                for val_index, (data, target) in enumerate(valloader):
                    data, label = Variable(data), Variable(target)
                    output = model(data)
                    loss = criterion(output, target)
                    losses.append(loss.data)
                    if val_index > 0 and val_index % 100 == 0:
                        break

                result = np.mean(losses)
                print("validation loss is {}".format(result))
                model.train()

if __name__ == '__main__':
    main()