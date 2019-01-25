#!/usr/local/bin/python3

from flask import Flask, render_template, request,url_for, jsonify
import os
import numpy as np
import sys
from flask_cors import CORS, cross_origin
import json
import matplotlib.pyplot as plt
import random
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import datetime
import data_generator as dg
import utils as utils
import skimage.color as color

app  = Flask(__name__)

app.debug = True

cwd = os.getcwd()

datasetname = 'mnist'

num_of_epoch  =100
num_of_nn  = 10

classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')


true_label = np.load(os.path.join(cwd,'static/data',datasetname,'true_label.npy'))

model_path =os.path.join(cwd,'static/data',datasetname,'model')

tsne_file  =os.path.join(cwd,'static/data',datasetname,'tsne_position.npy')

full_data_path =os.path.join(cwd,'static/data',datasetname,'data_full_layer')

bin_counts = np.insert(np.cumsum(np.bincount(true_label)) ,0,0)
sort_true_label = np.sort(true_label)
sort_index = np.argsort(true_label)


#Training_process_Comprision module starts###################################################
# The purposes of this module is as follows:
# 1. We obervsed the sudden jumping of accuracy for some classes, we want to know of this is
#    a common case for each of the 100 nns
# 2. If it happens in other nns, do they happen at the same epoch?
# 3. If we set up a threshold for the accuracy, what is the order of the classes being trained?
#     Isn't the same order for all the nns?

##########################################################################################
####data generate function################################################################
##output: a accuracy over all epochs of all nns , SHAPE: EPOCH X NN X LABEL_CLASS
def Training_process_data():
    bin_counts = None
    data_matrices = []


    for epoch in range(1, num_of_epoch+1):
        accuracy_data = []
        predict_label = []
        for nn in range(1,num_of_nn+1):
            last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[1]

        #predict_label = np.load(os.path.join(data_path,'predict_label_epoch{}.npy'.format(epoch)))
            predict_label.append(last_layer_data)

        #shape:num_of_nn X num_of_input
        predict_label =np.array(predict_label)
        #print(predict_label.shape)
        sort_predict_label = predict_label[:,sort_index]
        #print(sort_predict_label.shape)

        corrects = [ sort_predict_label[i]==sort_true_label for i in range(predict_label.shape[0]) ]
        corrects = np.array(corrects)
        #print(corrects)
        corrects_classes = np.array([np.sum(corrects[:,bin_counts[i]:bin_counts[i+1]],axis =1 )/(bin_counts[i+1]-bin_counts[i]) for i in range(10)])

        print(corrects_classes.shape)


        data_matrices.append(np.transpose(corrects_classes,(1,0)) )
        #print(data_matrices)

    return data_matrices

def TSNE_Caculator():
    TSNE_data = []
    for epoch in range(1, num_of_epoch+1):
        TSNE_epoch_data = []
        for nn in range(1,num_of_nn+1):
            last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[0]['o']
            tsne = TSNE(n_components=2)
            print("tNSE")
            trans_data = tsne.fit_transform(last_layer_data)
            print(trans_data.shape)
            TSNE_epoch_data.append(trans_data.tolist())
        TSNE_data.append(TSNE_epoch_data)

    return TSNE_data

#print("Here before")
accuracy_data=None
if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'accuracy_data.npy')):
    accuracy_data = np.load(os.path.join(cwd,'static/data',datasetname,'accuracy_data.npy'))
else:
    accuracy_data = np.array(Training_process_data()) #shape: num_of_epoch X number_of_nn X num_of_classes
    accuracy_data = np.transpose(accuracy_data,(1,0,2)) #shape: num_of_nn X number_of_epoch X num_of_classes
    np.save(os.path.join(cwd,'static/data',datasetname,'accuracy_data.npy'),accuracy_data)

# tsne_data = None
# if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'tsne_data.npy')):
#     tsne_data = np.load(os.path.join(cwd,'static/data',datasetname,'tsne_data.npy'))
# else:
#     tsne_data =TSNE_Caculator()
#     np.save(os.path.join(cwd,'static/data',datasetname,'tsne_data.npy'), tsne_data)




def jsonifydata():
    data_result = []
    average_result = []

    def jsonifyactualdata(data):
        data_new =[]
        for i,d in enumerate(data):
            data_new.append({'epoch': i+1})
            for j,v in enumerate(d):
                class_name = classes[j]
                data_new[i][class_name] = max(0.0,v)
        return data_new


    for nn_index in range(num_of_nn):
        accuracy_data_nn = accuracy_data[nn_index]
        diff_accuracy_data_nn = np.concatenate((np.reshape(accuracy_data_nn[0],(1,10)),np.diff(accuracy_data_nn,axis=0)),axis=0) #shape: num_epoch X number_of_classes
        average_result.append(diff_accuracy_data_nn)
        pass_data = jsonifyactualdata(diff_accuracy_data_nn)
        data_result.append({"nn_index": nn_index+1, "data": pass_data})

    average_result = np.array(average_result)
    average_result = np.mean(average_result,axis = 0)
    data_result.insert(0,{"nn_index": 0, "data": jsonifyactualdata(average_result)})
    #print(data_result[100])
    return data_result



@app.route('/', methods = ["GET", "POST"])
def index():
    data = jsonifydata()
    #dg.data_full_layer_vectors(1, 100)
    #tsne_data = np.load(tsne_file).tolist()
    # return render_template('index.html', data = data, num_of_nn = num_of_nn, num_of_epoch = num_of_epoch, tsne_data = tsne_data, true_label = true_label.tolist())
    return render_template('index.html', data = data, num_of_nn = num_of_nn, num_of_epoch = num_of_epoch)


@app.route('/data', methods =["GET", "POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_data():
    nn_idx,class_idx, epoch_idx = json.loads(request.get_data())
    nn_idx= int(nn_idx)
    class_idx=int(class_idx)
    epoch_idx= int(epoch_idx)
    print('nn_num:',nn_idx, 'class_idex:', class_idx, "epoch:" ,epoch_idx)
    if(nn_idx !=0 and epoch_idx>1):
        last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch_idx)))[0]['o']
        last_layer_data_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch_idx-1)))[0]['o']

        selected_index = sort_index[bin_counts[class_idx]:bin_counts[class_idx+1]]

        last_layer_diff = last_layer_data[selected_index] - last_layer_data_prev[selected_index]


        tsne_result = utils.TSNE_Caculator(last_layer_diff)

        loss_before = utils.SELoss_Caculator(last_layer_data_prev[selected_index],true_label[selected_index])


        loss_after = utils.SELoss_Caculator(last_layer_data[selected_index],true_label[selected_index])

        #print(loss_before,loss_after)

        # corrects = np.argmax(last_layer_data[selected_index], axis=1).tolist()

        # corrects_prev = np.argmax(last_layer_data_prev[selected_index], axis=1).tolist()
        # last_layer_data_prev =

        # true_labels = true_label[selected_index].tolist()
        colorArray = utils.Loss2RGB(loss_before, loss_after)[0].tolist()


        #print(colorArray)

        points_summary = {'position': tsne_result, 'colorArray': colorArray, 'index': selected_index.tolist()}

        # points_summary = {'position': tsne_result, 'correctness': corrects, 'correctness_prev': corrects_prev, 'index': selected_index.tolist(), 'true_label': true_labels}
        return jsonify(points_summary)



@app.route('/instance_data', methods=["GET","POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_image_data():
    data_summary = []
    nn_chosen,epoch_chosen,class_chosen,selectedDot = json.loads(request.get_data())
    nn_chosen = int(nn_chosen)
    epoch_chosen = int(epoch_chosen)
    class_chosen = int(class_chosen)
    selectedDot = int(selectedDot)
    print('nn_num: ',nn_chosen, ' class_idex: ', class_chosen, "epoch: " ,epoch_chosen, "Index:", selectedDot)

    data_file = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_chosen,epoch_chosen)))

    matrix_f1 = data_file[0]['f1'][selectedDot]

    matrix_f2  = data_file[0]['f2'][selectedDot]

    matrix_o = data_file[0]['o'][selectedDot]

    weight_f1 = data_file[2]['f1']

    weight_f2 = data_file[2]['f2']

    weight_o = data_file[2]['o']

    bias_f1 = data_file[3]['f1']

    bias_f2 = data_file[3]['f2']

    bias_o = data_file[3]['o']



    if(epoch_chosen>1):

        data_file_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_chosen,epoch_chosen-1)))

        matrix_f1_prev = data_file_prev[0]['f1'][selectedDot]

        matrix_f2_prev  = data_file_prev[0]['f2'][selectedDot]

        matrix_o_prev = data_file_prev[0]['o'][selectedDot]

        weight_f1_prev = data_file_prev[2]['f1']

        weight_f2_prev = data_file_prev[2]['f2']

        weight_o_prev = data_file_prev[2]['o']

        bias_f1_prev = data_file_prev[3]['f1']

        bias_f2_prev = data_file_prev[3]['f2']

        bias_o_prev = data_file_prev[3]['o']


        gradcam_prev = utils.GradientBackPropogation(nn_chosen,epoch_chosen-1,selectedDot)

        gradcam = utils.GradientBackPropogation(nn_chosen,epoch_chosen,selectedDot)

        gradcam_diff = (gradcam-gradcam_prev).tolist()




        data_summary.append({'label': 'f1', 'data_origin': matrix_f1.tolist(), 'data_prev': matrix_f1_prev.tolist(),'weight': weight_f1.tolist(), 'weight_prev': weight_f1_prev.tolist(), 'bias': bias_f1.tolist(), 'bias_prev': bias_f1_prev.tolist()})

        data_summary.append({'label': 'f2', 'data_origin': matrix_f2.tolist(), 'data_prev': matrix_f2_prev.tolist(),'weight': weight_f2.tolist(), 'weight_prev': weight_f2_prev.tolist(), 'bias': bias_f2.tolist(), 'bias_prev': bias_f2_prev.tolist()})

        data_summary.append({'label': 'o', 'data_origin': matrix_o.tolist(), 'data_prev': matrix_o_prev.tolist(),'weight': weight_o.tolist(), 'weight_prev': weight_o_prev.tolist(), 'bias': bias_o.tolist(), 'bias_prev': bias_o_prev.tolist()})

        data_summary.append(gradcam_diff)


    else:
        print("don't choose the first epoch")
    return jsonify(data_summary)













    # print(instance_data_matrix_f1.shape,instance_data_matrix_f2.shape,instance_data_matrix_o.shape,instance_data_weight_f1.shape,instance_data_weight_f2.shape,instance_data_weight_o.shape)








    # if nn_idx != 101 and epoch_idx>1: #101 means the overall
    #     selected_index = sort_index[bin_counts[class_idx]:bin_counts[class_idx+1]]

    #     matrix,_,weights = dg.data_full_layer_vectors(nn_idx,epoch_idx)
    #     matrix_prev,_,weights_prev = dg.data_full_layer_vectors(nn_idx,epoch_idx-1)

    #     matrix_c1 = matrix['c1'][selected_index].reshape(len(selected_index),-1)
    #     matrix_c1_prev =  matrix_prev['c1'][selected_index].reshape(len(selected_index),-1)
    #     matrix_c1_diff = matrix_c1 - matrix_c1_prev

    #     matrix_c2 = matrix['c2'][selected_index].reshape(len(selected_index),-1)
    #     matrix_c2_prev = matrix_prev['c2'][selected_index].reshape(len(selected_index),-1)
    #     matrix_c2_diff =  matrix_c2 - matrix_c2_prev

    #     matrix_f1 = matrix['f1'][selected_index].reshape(len(selected_index),-1)
    #     matrix_f1_prev = matrix_prev['f1'][selected_index].reshape(len(selected_index),-1)
    #     matrix_f1_diff =  matrix_f1 - matrix_f1_prev

    #     print(matrix_c1.shape)
    #     print(weights['c2'].shape)
    #     # f1_output_diff_rank = [np.argsort(np.mean(matrix_f1*weights['o'][i]-matrix_f1_prev*weights_prev['o'][i], axis=0))[::-1][0:10].tolist() for i in range(weights['o'].shape[0])]
    #     #print(f1_output_diff_rank)

    #     #f1_output_diff_rank = [np.argsort(np.mean(np.argsort((matrix_f1*weights['o'][i]-matrix_f1_prev*weights_prev['o'][i]), axis= 1)[::-1,::-1], axis=0))[0:10].tolist() for i in range(weights['o'].shape[0])]

    #     # print(f1_output_diff_rank)





    #     matrix_output = matrix['o'][selected_index].reshape(len(selected_index),-1)
    #     matrix_output_prev = matrix_prev['o'][selected_index].reshape(len(selected_index),-1)
    #     matrix_output_diff =  matrix_output - matrix_output_prev

    #     matrix_c1_diff =  np.mean(matrix_c1_diff,axis = 0)
    #     #matrix_c1_diff = matrix_c1_diff/np.amax(matrix_c1_diff)
    #     matrix_c2_diff = np.mean(matrix_c2_diff,axis = 0)
    #     #matrix_c2_diff = matrix_c1_diff/np.amax(matrix_c2_diff)
    #     matrix_f1_diff = np.mean(matrix_f1_diff,axis = 0)
    #     #matrix_f1_diff = matrix_f1_diff/np.amax(matrix_f1_diff)
    #     matrix_output_diff = np.mean(matrix_output_diff,axis = 0)
    #     #matrix_output_diff = matrix_output_diff/np.amax(matrix_output_diff)

    #     #print(matrix_c1_diff.shape)

    #     matrix_c1 =  np.mean(matrix_c1,axis = 0)
    #     matrix_c1_prev =  np.mean(matrix_c1_prev,axis = 0)
    #     matrix_c2 =  np.mean(matrix_c2,axis = 0)
    #     matrix_c2_prev =  np.mean(matrix_c2_prev,axis = 0)
    #     matrix_f1 =  np.mean(matrix_f1,axis = 0)
    #     matrix_f1_prev =  np.mean(matrix_f1_prev,axis = 0)
    #     matrix_output =  np.mean(matrix_output,axis = 0)
    #     matrix_output_prev =  np.mean(matrix_output_prev,axis = 0)

    #     data_summary=[]
    #     data_summary.append({'label': 'c1', 'data': matrix_c1_diff.tolist(),'data_origin': matrix_c1.tolist(), 'data_prev': matrix_c1_prev.tolist(),'weight': weights['c1'].tolist(), 'weight_prev': weights_prev['c1'].tolist()})
    #     data_summary.append({'label': 'c2', 'data': matrix_c2_diff.tolist(), 'data_origin': matrix_c2.tolist(), 'data_prev': matrix_c2_prev.tolist(),'weight': weights['c2'].tolist(), 'weight_prev': weights_prev['c2'].tolist()})
    #     data_summary.append({'label': 'f1', 'data': matrix_f1_diff.tolist(),'data_origin': matrix_f1.tolist(), 'data_prev': matrix_f1_prev.tolist(),'weight': weights['f1'].tolist(), 'weight_prev': weights_prev['f1'].tolist()})
    #     data_summary.append({'label': 'output', 'data': matrix_output_diff.tolist(),'data_origin': matrix_output.tolist(), 'data_prev': matrix_output_prev.tolist(),'weight': weights['o'].tolist(), 'weight_prev': weights_prev['o'].tolist()})
    #     return jsonify(data_summary)


    #dg.data_full_layer_vectors(1, 100)






if __name__ == '__main__':
    #print(true_label)
    app.run(host='0.0.0.0', port = 5000)

