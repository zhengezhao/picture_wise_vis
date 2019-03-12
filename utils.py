import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import createModel as createModel
import os
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from skimage import color
import torch.nn as nn
import torch.nn.functional as F
from math import *

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def cross_entropy(X,y):
    """
    X is a vector
    y is a label vector
    """
    m = len(X)
    p = stable_softmax(X)
    print(p)

    label = np.zeros(shape=(m,1))
    label[y] =1

    loss = log_loss(label,p)

    print(loss)
    return loss


classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

num_of_epoch = 100

cwd = os.getcwd()

datasetname = 'fashion-mnist'

images_folder = os.path.join(cwd,'static/data',datasetname,'test-images')
tsne_position = os.path.join(cwd,'static/data',datasetname,'tsne_position.npy')

def im_trans(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    image_trans = np.transpose(npimg, (1, 2, 0))
    return image_trans





def TSNE_Caculator(data):
    tsne = TSNE(n_jobs=6,n_components=2)
    trans_data = tsne.fit_transform(data)
    return trans_data.tolist()


def dumpImages():
    images = None
    labels = None
    for data in createModel.testloader_all:
        images, labels = data
        break
    for i in range(1000):
        image = images[i]
        fig = plt.figure(figsize=(1,1))
        axes  = fig.add_subplot(1,1,1)
        image_trans = im_trans(torchvision.utils.make_grid(image))
        axes.imshow(image_trans)
        plt.subplots_adjust(left=0,bottom=0 ,right=1, top=1)
        plt.axis('off')
        #fig.savefig('{}.png'.format(i))

        fig.savefig(os.path.join(images_folder,'{}.png'.format(i)))
        #fig.close()
        plt.close()


def dump_tsne_images():
    images = None
    labels=  None
    results =[]
    for data in createModel.testloader_all:
        images, labels = data
        break
    images = images.squeeze().numpy().reshape(1000,-1)
    labels = labels.numpy()
    #print(images.shape,labels)
    tsne = TSNE(n_jobs=6,n_components=2)
    trans_data = tsne.fit_transform(images)
    for i in range(len(labels)):
        results.append({'label': labels[i], 'data':trans_data[i].tolist(), 'index':i})
    print(results)
    np.save('tsne_position',results)

    #np.save('tsne_position',trans_data)
    #np.save()


# def plot_tsne_images():
#     data = np.load(tsne_position)
#     plt.scatter(data[:,0],data[:,1])
#     plt.show()


def SELoss_Caculator(data,true_label):



    data = np.array(data)
    #true_label = np.array(true_label)

    l = data.shape[0]

    d = data.shape[1]

    true_label_vectors = np.zeros(shape=(l,d))

    for i in range(l):
        true_label_vectors[i][true_label[i]] = 1

    return np.linalg.norm(data-true_label_vectors,axis =1)


def log_loss_Caculator(data,true_label):
    losses = []
    for i,data_ in enumerate(data):
        label = true_label[i]
        result = []
        for j,d in enumerate(data_):
            if j == label:
                result.append(log2(d))
            else:
                result.append(log2(1-d))
        losses.append(np.mean(result)*(-1.0))
    #print(type(losses))
    return losses








def Loss2RGB(loss_before, loss_after):
    loss_before = np.array(loss_before)
    loss_after  = np.array(loss_after)

    luminance  = (loss_before+ loss_after)/2

    L_value = np.interp(luminance,[np.amin(luminance), np.amax(luminance)],[0,100])

    a = (loss_after - loss_before)/2

    a_value = np.interp(a, [np.amin(a),np.amax(a)], [-128,128])

    b_value = [0.0 for i in range(len(luminance))]

    lab_values = list(zip(L_value,a_value,b_value))

    #print(lab_values)

    #print(color.lab2rgb([lab_values]))

    return color.lab2rgb([lab_values])*255.0



def loadNN(net,nnidx,epoch):
    if epoch == num_of_epoch:
        net_name = 'net_{}'.format(nnidx)
    else:
        net_name = 'net_{0}_epoch{1}'.format(nnidx,epoch)

    if os.path.isfile(os.path.join(createModel.model_path,net_name)):
        print(net_name+' is used.')
        net.load_state_dict(torch.load(os.path.join(createModel.model_path,net_name), map_location = 'cpu'))
    else:
        print(net_name + " is not trained")


def GradientBackPropogation(nnidx,epoch,index,label_index=None):

    if nnidx <3:
        net = createModel.initModel('fashion-mnist')
    else:
        net = createModel.initModel('fashion-mnist_2')

    # if nnidx ==1 or nnidx==3:
    #     optimizer= optim.SGD(net.parameters(), lr=0.00001, momentum=0.3)
    # else:
    #     optimizer= optim.SGD(net.parameters(), lr=0.00005, momentum=0.6)


    loadNN(net, nnidx, epoch)
    net.eval()

    input,label = createModel.testset[index]

    #print(input,label)




    input = input.unsqueeze_(0)

    input.requires_grad_(True)

    net.zero_grad()

    c1,c2,f1,output = net(input)

    #print(input)


    #print(c1.detach().numpy())

    #c1= c1.unsqueeze_(0)

    # c1.requires_grad_(True)

    #print(c1)
    #print(output)


    #input.requires_grad_(True)
    if label_index is not None:
        for i in range(10):
            if i not in label_index:
                output[0][i] = 0

    output= torch.norm(output[0])
    #print(output)

    output.backward()

    #print(c1.detach().numpy())
    #print(c1.grad)

    grad_of_param={}

    #print(input.grad.view(-1,28*28).numpy())

    # for name, parameter in net.named_parameters():
    #     grad_of_param[name] = parameter.grad.numpy()

    grad_of_param['input'] = input.grad.view(-1,28*28).numpy().flatten()

    input_new = input- input.grad

    net.zero_grad()

    c1_new,c2_new,f1_new,_ = net(input_new)

    grad_of_param['c1'] = c1_new.detach().numpy().flatten() - c1.detach().numpy().flatten()
    grad_of_param['c2'] = c2_new.detach().numpy().flatten() - c2.detach().numpy().flatten()
    grad_of_param['f1'] = f1_new.detach().numpy().flatten() - f1.detach().numpy().flatten()


    return grad_of_param






    # a = input.grad.view(-1,28*28).numpy()

    # return a




# def test():
#     x = torch.randn(3, requires_grad=True)

#     y = x * 2
#     while y.data.norm() < 1000:
#         y = y * 2

#     print(y)

#     v = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)
#     y.backward(v)

#     print(x.grad)


    # grad_of_param = {}
    # for name, parameter in net.named_parameters():
    #     if parameter.grad is not None:
    #         grad_of_param[name] = parameter.grad
    # print(grad_of_param)
    # return grad_of_param





    # optimizer_2.zero_grad()
    # _,_,output_2 = net_2(input)
    # loss_2 =  np.dot(output_2,o_data)

    # for index,data in enumerate(createModel.testset):
    #     #print(index)
    #     print(index,data)

    #     break







if __name__ == '__main__':
    #dumpImages()
    #dump_tsne_images()
    #plot_tsne_images()

    # cross_entropy(np.array([[1,0,0]]),np.array([[1,0,0]]))
    #print(log_loss([1,1],[[1,0],[0,1]]))

    #print(log_loss_Caculator([[0.8,0.1,0.1],[0.2,0.3,0.5]],[0,0]))

    #print(Loss2RGB([10,2,2,3,2,2,2],[3,5,6,7,8,9,6]))
    a = GradientBackPropogation(1,4,1,0)

    # print(a['input'].flatten().shape, a['c1'].flatten().shape, a['c2'].shape, a['f1'].shape)

    #b= GradientBackPropogation(1,3,1,0)['input']

    # print(b)
    # print(a)



    # plt.imshow((a-b).reshape(28,28))
    # plt.colorbar()
    # plt.show()



    #test()
