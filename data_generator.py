import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path
import numpy.linalg as linalg
import os
import createModel as createModel

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

num_of_nn = 10
num_of_epoch = 100

cwd = os.getcwd()

datasetname = 'mnist'
model_path =os.path.join(cwd,'static/data',datasetname,'model')

full_data_path =os.path.join(cwd,'static/data',datasetname,'data_full_layer')


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testloader_all = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=4)

testloader_one= torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=1)

trainloader_one = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, num_workers=1)

trainloader_all = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                              shuffle=False, num_workers=4)


def data_full_layer_vectors(nn_idx, epoch):

    images = None
    matrix = None
    predicts= None
    weights = None
    results =[]
###check if the results exists
    file_name = 'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch)

    if os.path.isfile(os.path.join(full_data_path,file_name)):
        results = np.load(full_data_path+'/'+file_name)
    else:

        for index,data in enumerate(testloader_all): ### just load it once
            images,_ = data
        net = createModel.Net()

        if epoch == num_of_epoch:
            net_name = 'net_{}'.format(nn_idx)
        else:
            net_name = 'net_{0}_epoch{1}'.format(nn_idx,epoch)

        if os.path.isfile(os.path.join(model_path,net_name)):
            print(net_name+' is used.')
            net.load_state_dict(torch.load(os.path.join(model_path,net_name), map_location = 'cpu'))
        else:
            print(net_name + " is not trained")


        with torch.no_grad():
            net.eval()
            f1,f2,outputs = net(images)  ##outputs shape : number_of_inputs X 10
            _,predicted = torch.max(outputs,1)
            #print(outputs.shape)
            m = nn.Softmax(dim=1)
            outputs = m(outputs)
            matrix  ={'f1':f1.numpy(), 'f2': f2.numpy(), 'o': outputs.numpy()}
            predicts = predicted.numpy()
            weights = {'f1': net.fc1[0].weight.detach().numpy(),'f2': net.fc2[0].weight.detach().numpy(), 'o': net.fc3[0].weight.detach().numpy()}

        results.append(matrix)
        results.append(predicts)
        results.append(weights)


        np.save(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch)),results)

    return  results




if __name__ == '__main__':
    for i in range(num_of_nn):
        for j in range(num_of_epoch):
            data_full_layer_vectors(i+1, j+1)




