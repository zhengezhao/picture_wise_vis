import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self,num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size =2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size =2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(128,10))



    def forward(self, x):
        out = self.conv1(x)
        c1 = out
        out = self.conv2(out)
        c2= out
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        f1 = out
        out = self.fc2(out)
        return c1,c2,f1,out
