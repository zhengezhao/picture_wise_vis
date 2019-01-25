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

datasetname = 'mnist'

images_folder = os.path.join(cwd,'static/data',datasetname,'test-images')
tsne_position = os.path.join(cwd,'static/data',datasetname,'tsne_position.npy')

def im_trans(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    image_trans = np.transpose(npimg, (1, 2, 0))
    return image_trans



def Show_output_layer(netname, file_address):
    if os.path.isfile(os.path.join(file_address,netname)):
        print(netname + " read from the disk")
        net = createModel.Net()
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
        for data in createModel.testloader_all:
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


def TSNE_Caculator(data):
    tsne = TSNE(n_jobs=6,n_components=2)
    trans_data = tsne.fit_transform(data)
    return trans_data.tolist()


def dumpImages():
    for data in createModel.testloader_all:
        images, labels = data
        for i in range(10000):
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


def dump_tnse_images():
    for data in createModel.testloader_all:
        images, labels = data
        images = images.squeeze().numpy().reshape(10000,-1)
        tsne = TSNE(n_jobs=6,n_components=2)
        trans_data = tsne.fit_transform(images)
        np.save('tsne_position',trans_data)


def plot_tnse_images():
    data = np.load(tsne_position)
    plt.scatter(data[:,0],data[:,1])
    plt.show()


def SELoss_Caculator(data,true_label):

    data = np.array(data)
    true_label = np.array(true_label)

    l = data.shape[0]

    d = data.shape[1]

    true_label_vectors = np.zeros(shape=(l,d))

    for i in range(l):
        true_label_vectors[i][true_label[i]] = 1

    return np.linalg.norm(data-true_label_vectors,axis =1)



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



def GradientBackPropogation(nnidx,epoch,index):

    net = createModel.Net()

    optimizer= optim.SGD(net.parameters(), lr=0.00001, momentum=0.3)
    loadNN(net, nnidx, epoch)

    input,label = createModel.testset[index]

    input.requires_grad_(True)

    optimizer.zero_grad()

    _,_,output = net(input)

    output = torch.norm(output)

    output.backward()


    a = input.grad.view(-1,28*28).numpy()

    return a



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
    #dump_tnse_images()
    #plot_tnse_images()

    # cross_entropy(np.array([[1,0,0]]),np.array([[1,0,0]]))
    #print(log_loss([1,1],[[1,0],[0,1]]))

    #print(SELoss_Caculator([[1,0,0],[1,0,0]],[0,2]))

    #print(Loss2RGB([10,2,2,3,2,2,2],[3,5,6,7,8,9,6]))
    GradientBackPropogation(1,1,1)
    #test()
