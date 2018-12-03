import torch
import torchvision
import torchvision.transforms as transforms
import createModel as createModel
import os
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt


classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

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




if __name__ == '__main__':
    #dump_tnse_images()
    plot_tnse_images()






