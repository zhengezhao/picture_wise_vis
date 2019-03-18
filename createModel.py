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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_cuda = False
num_of_nn = 2
num_of_epoch=100
batch_size = 256
test_batch_size = 64
classes,transform,Dataset  = None, None, None
trainloader, testloader = None, None
trainset, testset = None, None
kwargs = {}

cwd = os.getcwd()
datasetname = 'fashion-mnist'

if datasetname == 'mnist':
    Dataset = torchvision.datasets.MNIST
    classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])


elif datasetname == 'fashion-mnist':
    Dataset = torchvision.datasets.FashionMNIST
    classes = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')
    transform_mean = (0.1307,)
    transform_std = (0.3081,)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])

elif datasetname == 'cifar10':
    Dataset = torchvision.datasets.CIFAR10
    classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
    transform_mean = (0.5, 0.5, 0.5)
    transform_std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])
else:
    print("The dataset doesn't exit")
    sys.exit()


trainset = Dataset('./dataset/'+datasetname, train=True, download=True, transform=transform)
testset = Dataset('./dataset/'+datasetname, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(
        trainset,batch_size=batch_size, shuffle=True,**kwargs)

testloader = torch.utils.data.DataLoader(
        testset,batch_size=test_batch_size, shuffle=False,**kwargs)

testloader_all = torch.utils.data.DataLoader(
        Dataset('./dataset/'+datasetname, train=False, transform=transform),
            batch_size=1000, shuffle=False,
            **kwargs)

testloader_one = torch.utils.data.DataLoader(
        Dataset('./dataset/'+datasetname, train=False, transform=transform),
            batch_size=1, shuffle=False,
            **kwargs)


model_path =os.path.join(cwd,'static/data',datasetname,'model')


def TrainNN(net, net_name, lr = None , momentum=None):
     # set up the loss function momentum to avoid local optimal
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss(size_average = False)
    #criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters())
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    net.to(device)

    if os.path.isfile(os.path.join(model_path,net_name)):
        print(net_name + "has been trained.")
        net.load_state_dict(torch.load(os.path.join(model_path,net_name), map_location = 'cpu'))


    else:

        net_name_epoch = net_name+'_epoch0'
        torch.save(net.state_dict(), os.path.join(model_path,net_name_epoch))

    # train the network
        for epoch in range(num_of_epoch):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)[-1]
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
            outputs = net(images)[-1]
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

def initModel(modelname, use_cuda=use_cuda):
    if modelname == 'mnist':
        from model_mnist import Net
    elif modelname == 'cifar10':
        from model_cifar10 import Net
    elif modelname == 'fashion-mnist':
        from model_fashion_mnist import Net
    elif modelname == 'fashion-mnist_2':
        from model_fashion_mnist_2 import Net
    if use_cuda:
        model = Net().cuda()
    else:
        model = Net()
    return model


def Show_output_layer(file_address, model_name, netname):

    net = initModel(model_name)

    if os.path.isfile(os.path.join(file_address,netname)):
        print(netname + " read from the disk")

        net.load_state_dict(torch.load(os.path.join(file_address,netname), map_location = 'cpu'))
    else:
        print("not trained nn " + netname)
        return

    ###Show the accuracy
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    net.eval()
    with torch.no_grad():
        for data in testloader_all:
            images, labels = data
            outputs = net(images)[-1]
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


if __name__ == '__main__':
    # torch.manual_seed(0)

    # net = initModel('fashion-mnist')
    # net.to(device)
    # net_name_epoch = 'net_2_epoch0'
    # torch.save(net.state_dict(), os.path.join(model_path,net_name_epoch))

    # Show_output_layer(model_path,'fashion-mnist','net_1_epoch1')
    # Show_output_layer(model_path,'fashion-mnist','net_2_epoch1')
    # Show_output_layer(model_path,'fashion-mnist_2','net_3_epoch1')
    # Show_output_layer(model_path,'fashion-mnist_2','net_4_epoch1')

    net = initModel('fashion-mnist')

    TrainNN(net, 'net_6', lr = 0.00001, momentum=0.3)










