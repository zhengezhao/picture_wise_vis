import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        layers = [nn.Conv2d(1, 10, 5), nn.MaxPool2d(2), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)

        layers = [nn.Conv2d(10, 20, 5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)

        layers = [nn.Linear(320, 50), nn.ReLU()]
        self.fc1 = nn.Sequential(*layers)

        self.drop = nn.Dropout(p=0.2)

        layers = [nn.Linear(50, 10), nn.ReLU()]
        self.fc2 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        c1 = x
        c1.retain_grad()
        x = self.conv2(x)
        c2 = x
        c2.retain_grad()
        x = x.view(-1, 320)
        x = self.fc1(x)
        f1 = x
        f1.retain_grad()
        x = self.drop(x)
        x = self.fc2(x)
        return c1,c2,f1,x
