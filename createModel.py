import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path
import numpy.linalg as linalg
import os
from scipy.stats import ortho_group
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm
from sklearn.decomposition import PCA
from matplotlib import colors



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)



trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=5)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=5)


testloader_all = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=4)

classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_of_nn = 10
num_of_epoch=100


cwd = os.getcwd()
datasetname = 'mnist'
model_path =os.path.join(cwd,'static/data',datasetname,'model')


def TrainNN(net, net_name):
     # set up the loss function momentum to avoid local optimal
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss(size_average = False)
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.3)
    net.to(device)

    if os.path.isfile(os.path.join(model_path,net_name)):
        print(net_name + "has been trained.")
        return


    # train the network
    for epoch in range(num_of_epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs, labels = inputs.to(device), labels.to(device)
            _,_,outputs = net(inputs)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 125 == 124:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        #save the intermedial model in each epoch
        if epoch != num_of_epoch - 1:
            net_name_epoch = net_name + '_epoch' + str(epoch + 1)
        else:
            net_name_epoch = net_name
        torch.save(net.state_dict(), os.path.join(model_path,net_name_epoch))

    print('Finished Training {}'.format(net_name))

    ###Show the accuracy
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            _,_, outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


# F=1, PAD=0; F=3, PAD=1; F=5, PAD=2; F=7, PAD=3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        a = nn.Conv2d(1, 10, 5)
        layers = [nn.Conv2d(1, 10, 5), nn.MaxPool2d(2), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)
        layers = [nn.Conv2d(10, 20, 5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)
        layers = [nn.Linear(28*28, 320), nn.ReLU()]
        self.fc1 = nn.Sequential(*layers)
        layers = [nn.Linear(320, 50), nn.ReLU()]
        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Sequential(*layers)
        layers = [nn.Linear(50, 10), nn.ReLU()]
        self.fc3 = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        f1 = x
        x = self.drop(x)
        x = self.fc2(x)
        f2 = x
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return f1,f2,x




if __name__ == '__main__':
    for i in range(10):
        net = Net()
        net_name = 'net_{}'.format(str(i+1))
        TrainNN(net,net_name)

