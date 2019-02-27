import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        layers = [nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)]
        self.conv1 = nn.Sequential(*layers)
        
        layers = [nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)]
        self.conv2 = nn.Sequential(*layers)
        
        layers = [nn.Linear(16 * 5 * 5, 120), nn.ReLU()]
        self.fc1 = nn.Sequential(*layers)
        self.drop1 = nn.Dropout(0.1)
        
        layers = [nn.Linear(120, 84), nn.ReLU()]
        self.fc2 = nn.Sequential(*layers)
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x