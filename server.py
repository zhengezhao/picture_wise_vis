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
import createModel as createModel

app  = Flask(__name__)

app.debug = True

cwd = os.getcwd()

datasetname = 'fashion-mnist'

num_of_epoch  =100
num_of_nn  = 2

classes = createModel.classes


true_label = [createModel.testset[i][1].numpy().tolist() for i in range(1000)]
# print(true_label)

model_path =os.path.join(cwd,'static/data',datasetname,'model')


full_data_path =os.path.join(cwd,'static/data',datasetname,'data_full_layer')

bin_counts = np.insert(np.cumsum(np.bincount(true_label)) ,0,0)
sort_true_label = np.sort(true_label)
sort_index = np.argsort(true_label)

#print(bin_counts)
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

        #print(corrects_classes.shape)


        data_matrices.append(np.transpose(corrects_classes,(1,0)) )
        #print(data_matrices)

    return data_matrices


def Loss_Data_Change():
    data_matrices = []
    for epoch in range(0, num_of_epoch+1):
        accuracy_data = []
        losses= []
        for nn in range(1,num_of_nn+1):
            last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[0]['o']

        #predict_label = np.load(os.path.join(data_path,'predict_label_epoch{}.npy'.format(epoch)))
            losses.append(last_layer_data)

        #shape:num_of_nn X num_of_input X 10
        losses = np.array(losses)
        #print(losses.shape)
        sort_losses = losses[:,sort_index]
        #print(sort_losses.shape)

        losses_sum = [utils.log_loss_Caculator(sort_losses[i],sort_true_label) for i in range(sort_losses.shape[0]) ]

        losses_sum = np.array(losses_sum)
        #print(losses_sum.shape)

        loss_classes = np.array([np.sum(losses_sum[:,bin_counts[i]:bin_counts[i+1]],axis =1 ) for i in range(10)])

        #print(loss_classes.shape)


        data_matrices.append(np.transpose(loss_classes,(1,0)) )
   # print(data_matrices)

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


def dumpPredictions():
    predictions = []
    for epoch in range(1, num_of_epoch+1):
        predict_label = []
        for nn in range(1,num_of_nn+1):
            last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[1]
            predict_label.append(last_layer_data)

        #shape:num_of_nn X num_of_input
        predict_label =np.array(predict_label)
        #print(predict_label.shape)
        corrects = [ predict_label[i]==np.array(true_label) for i in range(predict_label.shape[0]) ]
        predictions.append(corrects)
    predictions = np.array(predictions)
    predictions = np.transpose(predictions,(2,1,0))  #shape: num_of_instances X number_of_nn X num_of_epoch
    print(predictions.shape)
    return predictions

def dumpLosses():
    losses = []
    for epoch in range(0, num_of_epoch+1):
        outputs_epoch = []
        for nn in range(1,num_of_nn+1):
            last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[0]['o']
            outputs_epoch.append(last_layer_data)

        #shape:num_of_nn X num_of_input X 10
        outputs_epoch =np.array(outputs_epoch)
        #print(predict_label.shape)
        losses_epoch = [ utils.log_loss_Caculator(outputs_epoch[i],np.array(true_label)) for i in range(outputs_epoch.shape[0]) ]
        losses.append(losses_epoch)
    losses = np.array(losses)
    losses = np.transpose(losses,(2,1,0))  #shape: num_of_instances X number_of_nn X num_of_epoch
    print(losses.shape)
    return losses


# #print("Here before")
# accuracy_data=None
# if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'accuracy_data.npy')):
#     accuracy_data = np.load(os.path.join(cwd,'static/data',datasetname,'accuracy_data.npy'))
# else:
#     accuracy_data = np.array(Training_process_data()) #shape: num_of_epoch X number_of_nn X num_of_classes
#     accuracy_data = np.transpose(accuracy_data,(1,0,2)) #shape: num_of_nn X number_of_epoch X num_of_classes
#     np.save(os.path.join(cwd,'static/data',datasetname,'accuracy_data.npy'),accuracy_data)
loss_data = None
if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'loss_data.npy')):
    loss_data = np.load(os.path.join(cwd,'static/data',datasetname,'loss_data.npy'))
else:
    loss_data = np.array(Loss_Data_Change()) #shape: num_of_epoch X number_of_nn X num_of_classes
    loss_data = np.transpose(loss_data,(1,0,2)) #shape: num_of_nn X number_of_epoch X num_of_classes
    np.save(os.path.join(cwd,'static/data',datasetname,'loss_data.npy'),loss_data)


tsne_data = None
if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'tsne_position.npy')):
    tsne_data = np.load(os.path.join(cwd,'static/data',datasetname,'tsne_position.npy'))
else:
    print("The tsne data is missing")


losses_instance_data = None
if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'losses_instance_data.npy')):
    losses_instance_data = np.load(os.path.join(cwd,'static/data',datasetname,'losses_instance_data.npy'))
else:
    losses_instance_data  = dumpLosses()
    np.save(os.path.join(cwd,'static/data',datasetname,'losses_instance_data.npy'),losses_instance_data)


predictions_data = None
if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'predictions_data.npy')):
    predictions_data = np.load(os.path.join(cwd,'static/data',datasetname,'predictions_data.npy'))
else:
    predictions_data  = dumpPredictions()
    np.save(os.path.join(cwd,'static/data',datasetname,'predictions_data.npy'),predictions_data)


    # tsne_data =TSNE_Caculator()
    # np.save(os.path.join(cwd,'static/data',datasetname,'tsne_data.npy'), tsne_data)



def jsonifydata(origin_data):
    data_result = []
    # average_result = []

    def jsonifyactualdata(data):
        data_new =[]
        for i,d in enumerate(data):
            data_new.append({'epoch': i+1})
            for j,v in enumerate(d):
                class_name = classes[j]
                #data_new[i][class_name] = max(0.0,v)#for streamgraph
                data_new[i][class_name] = v#for stacked bar chart
        return data_new


    for nn_index in range(num_of_nn):
        origin_data_nn = origin_data[nn_index]
        diff_data_nn = np.diff(origin_data_nn,axis=0) #shape: num_epoch X number_of_classes
        print(diff_data_nn.shape)
        # average_result.append(diff_data_nn)
        pass_data = jsonifyactualdata(diff_data_nn)
        data_result.append({"nn_index": nn_index+1, "data": pass_data})

    # average_result = np.array(average_result)
    # average_result = np.mean(average_result,axis = 0)
    # data_result.insert(0,{"nn_index": 0, "data": jsonifyactualdata(average_result)})
    #print(data_result[100])
    return data_result


@app.route('/', methods = ["GET", "POST"])
def index():
    data = jsonifydata(loss_data)
    #dg.data_full_layer_vectors(1, 100)
    #tsne_data = np.load(tsne_file).tolist()
    # return render_template('index.html', data = data, num_of_nn = num_of_nn, num_of_epoch = num_of_epoch, tsne_data = tsne_data, true_label = true_label.tolist())
#print(type(tsne_data.tolist()))
    #print(type(data))
    return render_template('index.html', data = data, num_of_nn = num_of_nn, num_of_epoch = num_of_epoch, classes = list(classes), tsne_data= tsne_data.tolist())


@app.route('/class_data', methods =["GET", "POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_class_data():
    epoch_idx,class_name,modelid = json.loads(request.get_data())
    class_idx = classes.index(class_name)
    model_idx = int(modelid[5:])
    epoch_idx = int(epoch_idx)
    selected_index = sort_index[bin_counts[class_idx]:bin_counts[class_idx+1]]
    print('epoch:',epoch_idx,'class:',class_name,'class_idx:',class_idx,'model_idx:',model_idx)
    #print(seltecte)

    # predicts_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(model_idx,epoch_idx)))[1]
    # predicts_data_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(model_idx,epoch_idx-1)))[1]
    #print(last_layer_data.shape,last_layer_data_prev.shape,predicts_data.shape,predicts_data_prev.shape)

        #this is the index which is predicted different from the true class previously  or currently
    # diff_indx=[]
    # for indx in selected_index:
    #     if (predicts_data[indx] == class_idx and predicts_data_prev[indx] != class_idx) or (predicts_data[indx] != class_idx and predicts_data_prev[indx] == class_idx):
    #         diff_indx.append(int(indx))

    #print(last_layer_data_prev[diff_indx],last_layer_data[diff_indx],np.array(true_label)[diff_indx])


    #shape of losses_instance_data: num_of_instances X number_of_nn X num_of_epoch

    print(losses_instance_data.shape)

    print(losses_instance_data[selected_index].shape)

    loss_after = losses_instance_data[selected_index][:,model_idx-1,epoch_idx]

    loss_before = losses_instance_data[selected_index][:,model_idx-1,epoch_idx-1]

    #print(loss_before,loss_after)

    points_summary = {'whole_index':selected_index.tolist(), 'loss_before': loss_before.tolist(), 'loss_after': loss_after.tolist()}


    return jsonify(points_summary)



@app.route('/loss_sub_data', methods =["GET", "POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_accuracy_data():
    brushed_index  = json.loads(request.get_data())
    print(brushed_index)


    overall_data = np.transpose(losses_instance_data[brushed_index],(1,2,0)) #shape: nn X epoch X instances

    sub_true_label = np.array(true_label)[brushed_index]

    label_counts = np.zeros(10)

    for label in sub_true_label:
        label_counts[label] +=1

    sort_true_label = np.sort(sub_true_label)
    sort_index = np.argsort(sub_true_label)

    sub_bin_counts = np.insert(np.cumsum(label_counts),0,0).astype(int)
    print(bin_counts)

    sort_losses =overall_data[:,:,sort_index]

    results = []
    for nn in range(num_of_nn):
        data1  = overall_data[nn] #shape: epoch X instances
        data = []
        for e in range(1,num_of_epoch+1):
            data_class = {}
            data_class['epoch'] = e
            for i,c in enumerate(classes):
                if label_counts[i]!=0:
                    #print(sort_predict_label[nn][e],bin_counts[i], bin_counts[i+1])
                    sum_after = np.sum(sort_losses[nn][e][sub_bin_counts[i]:sub_bin_counts[i+1]])
                    sum_prev = np.sum(sort_losses[nn][e-1][sub_bin_counts[i]:sub_bin_counts[i+1]])
                    data_class[c] = (sum_after - sum_prev)
                else:
                    data_class[c] = 0
            data.append(data_class)
        results.append({'nn_index':nn+1, 'data': data})
    #print(results)

    return jsonify(results)









    # sum_corrects = np.sum(np.transpose(predictions_data[brushed_index],(1,2,0)), axis= 2) / len(brushed_index)

    # correctes_diff =np.concatenate((sum_corrects[:,0].reshape(1,-1).T,np.diff(sum_corrects,axis =1)),axis=1)


    # return jsonify(correctes_diff.tolist())





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
    epochs =[]
    epoch1_chosen,epoch2_chosen,selectedDot = json.loads(request.get_data())
    epochs.append(int(epoch1_chosen))
    epochs.append(int(epoch2_chosen))
    print('epoch for model 1: ',epochs[0],"epoch for model 2 " ,epochs[1], "Index:", selectedDot)

    for i,nn_chosen in enumerate(range(1,3)):

        data_s  =[]

        epoch_chosen = epochs[i]

        data_file = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_chosen,epoch_chosen)))

        data_file_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_chosen,epoch_chosen-1)))

        gradcam_prev = utils.GradientBackPropogation(nn_chosen,epoch_chosen-1,selectedDot)

        gradcam = utils.GradientBackPropogation(nn_chosen,epoch_chosen,selectedDot)
        gradcam_diff={}

        for name in gradcam.keys():
            gradcam_diff[name]= (gradcam[name] - gradcam_prev[name]).tolist()



        matrix_f1 = data_file[0]['f1'][selectedDot]

        matrix_o = data_file[0]['o'][selectedDot]

        #print(sum(matrix_o))

        matrix_c1 = data_file[0]['c1'][selectedDot].flatten()


        matrix_c2 = data_file[0]['c2'][selectedDot].flatten()

        print(matrix_c2.shape,matrix_c1.shape,matrix_f1.shape, matrix_o.shape)

        matrix_f1_prev = data_file_prev[0]['f1'][selectedDot]

        matrix_o_prev = data_file_prev[0]['o'][selectedDot]
        #print(sum(matrix_o_prev))

        matrix_c1_prev = data_file_prev[0]['c1'][selectedDot].flatten()

        matrix_c2_prev = data_file_prev[0]['c2'][selectedDot].flatten()

        data_s.append({'label': 'c1', 'data_origin': matrix_c1.tolist(), 'data_prev': matrix_c1_prev.tolist()})
        data_s.append({'label': 'c2', 'data_origin': matrix_c2.tolist(), 'data_prev': matrix_c2_prev.tolist()})
        data_s.append({'label': 'f1', 'data_origin': matrix_f1.tolist(), 'data_prev': matrix_f1_prev.tolist()})
        data_s.append({'label': 'o', 'data_origin': matrix_o.tolist(), 'data_prev': matrix_o_prev.tolist()})



        data_s.append(gradcam_diff)

        data_summary.append(data_s)



    return jsonify(data_summary)
    #     gradcam_prev = utils.GradientBackPropogation(nn_chosen,epoch_chosen-1,selectedDot)

    #     gradcam = utils.GradientBackPropogation(nn_chosen,epoch_chosen,selectedDot)

    #     gradcam_diff = (gradcam-gradcam_prev).tolist()




    #     data_summary.append({'label': 'f1', 'data_origin': matrix_f1.tolist(), 'data_prev': matrix_f1_prev.tolist(),'weight': weight_f1.tolist(), 'weight_prev': weight_f1_prev.tolist(), 'bias': bias_f1.tolist(), 'bias_prev': bias_f1_prev.tolist()})

    #     data_summary.append({'label': 'f2', 'data_origin': matrix_f2.tolist(), 'data_prev': matrix_f2_prev.tolist(),'weight': weight_f2.tolist(), 'weight_prev': weight_f2_prev.tolist(), 'bias': bias_f2.tolist(), 'bias_prev': bias_f2_prev.tolist()})

    #     data_summary.append({'label': 'o', 'data_origin': matrix_o.tolist(), 'data_prev': matrix_o_prev.tolist(),'weight': weight_o.tolist(), 'weight_prev': weight_o_prev.tolist(), 'bias': bias_o.tolist(), 'bias_prev': bias_o_prev.tolist()})

    #     data_summary.append(gradcam_diff)


    # else:
    #     print("don't choose the first epoch")
    # return jsonify(data_summary)













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

