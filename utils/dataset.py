import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class Mydataset(Dataset):
    def __init__(self, txt_path, transform):
        super(Dataset).__init__()
        input = open(txt_path, "r")
        lines = input.readlines()
        input.close()
        data = []
        for line in lines:
            path = line.split()[0]
            label = int(line.split()[1])
            data.append([path, label])
        self.data = data
        self.transform = transform
    def __getitem__(self, index):
        path, label = self.data[index]
        image =  Image.open(path)
        #image = cv2.imread(path)
        image = self.transform(image)
        return image, label-1
    def __len__(self):
        return len(self.data)

def alexnet_transform():
    transform = transforms.Compose([
                                    transforms.Resize((227,227)),
                                    transforms.ToTensor(),
                                    ])
    return transform

def googlenet_transform():
    #the original googlenet crops 224*224 image from right,left, center 
    # sides of images. Here I just resize the image to be 227*227
    transform = transforms.Compose([
                                    transforms.Resize((227,227)),
                                    transforms.ToTensor(),
                                    ])
    return transform

def resnet_transform():
    #the original googlenet crops 224*224 image from right,left, center 
    # sides of images. Here I just resize the image to be 227*227
    transform = transforms.Compose([
                                    transforms.Resize((227,227)),
                                    transforms.ToTensor(),
                                    ])
    return transform
