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

num_of_nn = 4
num_of_epoch = 100

cwd = os.getcwd()

datasetname = 'fashion-mnist'
model_path =os.path.join(cwd,'static/data',datasetname,'model')

full_data_path =os.path.join(cwd,'static/data',datasetname,'data_full_layer')



def data_full_layer_vectors(modelname,nn_idx, epoch):

    images = None
    matrix = None
    predicts= None
    weights = None
    labels= None
    results =[]
###check if the results exists
    file_name = 'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch)

    if os.path.isfile(os.path.join(full_data_path,file_name)):
        results = np.load(full_data_path+'/'+file_name)
    else:
        net = createModel.initModel(modelname)



        for index,data in enumerate(createModel.testloader_all): ### just load it once
            images,_ = data
            break

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
            c1,c2,f1,outputs = net(images)  ##outputs shape : number_of_inputs X 10
            _,predicted = torch.max(outputs,1)
            #print(outputs.shape)
            m = nn.Softmax(dim=1)
            outputs = m(outputs)
            matrix  ={'c1':c1.numpy(), 'c2': c2.numpy(), 'f1': f1.numpy(), 'o': outputs.numpy()}
            predicts = predicted.numpy()
            # weights = {'f1': net.fc1[0].weight.detach().numpy(),'f2': net.fc2[0].weight.detach().numpy(), 'o': net.fc3[0].weight.detach().numpy()}
            # bias = {'f1': net.fc1[0].bias.detach().numpy(),'f2': net.fc2[0].bias.detach().numpy(), 'o': net.fc3[0].bias.detach().numpy()}
            #print(bias['f1'].shape, bias['f2'].shape,bias['o'].shape)


        results.append(matrix)
        results.append(predicts)
        # results.append(weights)
        # results.append(bias)


        np.save(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch)),results)

    return  results




if __name__ == '__main__':
    data_full_layer_vectors('fashion-mnist',1, 0)
    data_full_layer_vectors('fashion-mnist',2, 0)
    data_full_layer_vectors('fashion-mnist_2',3, 0)
    data_full_layer_vectors('fashion-mnist_2',4, 0)
    # for i in range(num_of_nn
    #     for j in range(num_of_epoch):
    #         data_full_layer_vectors(i+1, j+1)




