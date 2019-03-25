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
num_of_nn  = [5,6]


classes = createModel.classes

classes_pos_neg  = []

for c in classes:
    classes_pos_neg.append(c+'(+)')
for c in classes:
    classes_pos_neg.append(c+'(-)')

num_of_input = 1000
#print(classes_pos_neg)

true_label = [createModel.testset[i][1].numpy().tolist() for i in range(num_of_input)]
# print(true_label)

model_path =os.path.join(cwd,'static/data',datasetname,'model')


full_data_path =os.path.join(cwd,'static/data',datasetname,'data_full_layer')

bin_counts = np.insert(np.cumsum(np.bincount(true_label)) ,0,0)
sort_true_label = np.sort(true_label)
sort_index = np.argsort(true_label)



def dumpLosses():
    losses = []
    for epoch in range(0, num_of_epoch+1):
        outputs_epoch = []
        for nn in num_of_nn:
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

def dumpPredictions():
    predictions =[]
    for epoch in range(0,num_of_epoch+1):
        outputs_epoch=[]
        for nn in num_of_nn:
            prediciton_file = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[1]
            outputs_epoch.append(prediciton_file)
        predictions.append(outputs_epoch)

    predictions= np.array(predictions)
    predictions = np.transpose(predictions,(2,1,0)) #shape: num_of_instances X number_of_nn X num_of_epoch
    #print(predictions.shape)
    return predictions




predictions_data = None
if os.path.isfile(os.path.join(cwd,'static/data',datasetname,'predictions_data.npy')):
    predictions_data = np.load(os.path.join(cwd,'static/data',datasetname,'predictions_data.npy'))
else:
    predictions_data  = dumpPredictions()
    np.save(os.path.join(cwd,'static/data',datasetname,'predictions_data.npy'),predictions_data)

print(predictions_data.shape)

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



def Loss_Difference_Summary(input_data,bin_counts,sort_index):
    def classifydata(data):
        data_json = []
        for i,instances in enumerate(data):
            data_json.append({'epoch':i+1})
            pos_loss_classes = [np.sum(instances[bin_counts[i]:bin_counts[i+1]].clip(min=0)) for i in range(10)]
            neg_loss_classes = [np.sum(instances[bin_counts[i]:bin_counts[i+1]].clip(max=0)) for i in range(10)]
            loss_classes = pos_loss_classes+neg_loss_classes
            for j,v in enumerate(loss_classes):
                class_name = classes_pos_neg[j]
                data_json[i][class_name] = v
        return data_json


    #shape of losses_instance_data: num_of_instances X number_of_nn X num_of_epoch+1
    diff_data = np.diff(input_data, axis=2) #shape: num_of_instances X number_of_nn X num_of_epoch
    data_result =[]
    diff_data = np.transpose(diff_data,(1,2,0)) #shape: number_of_nn X num_of_epoch X num_of_instances
    diff_data = diff_data[:,:,sort_index]
    for nn_index in range(len(num_of_nn)):
        diff_data_nn = diff_data[nn_index] #shape : num_of_epoch X num_of_instance
        pass_data = classifydata(diff_data_nn)
        data_result.append({"nn_index": num_of_nn[nn_index], "data": pass_data})
    return data_result


def DivergingInstancesFinder(top_num):
    outputs = []
    for epoch in range(0, num_of_epoch+1):
        outputs_epoch = []
        for nn in num_of_nn:
            last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[0]['o']
            outputs_epoch.append(last_layer_data)
        outputs.append(outputs_epoch)
    outputs= np.array(outputs) #shape: num_of_epoch X num_of_nn X num_of_instances X output

    outputs = np.transpose(outputs,(2,1,0,3)) #shape: num_of_instances X num_of_nn X num_of_epoch X output


    print(outputs.shape)
    diffs=[]

    for instance in outputs:
        v1 = np.array([utils.softmax(instance[0][i]) for i in range(num_of_epoch+1)])
        #print(v1.shape)
        v1 = v1.flatten()
        v2 = np.array([utils.softmax(instance[1][i]) for i in range(num_of_epoch+1)]).flatten()
        diff = np.linalg.norm(v1-v2)
        diffs.append(diff)

    diffs =np.array(diffs)

    sorted_diffs = np.sort(diffs)[::-1]
    sorted_indices = np.argsort(diffs)[::-1]

    return sorted_indices[:top_num]


def DivergingInstancesFinder2(top_num):
    diffs = []
    for instance in losses_instance_data: #num_of_instances X number_of_nn X num_of_epoch
        v1 = instance[0]
        v2=  instance[1]
        diff = np.linalg.norm(v1-v2)
        diffs.append(diff)
    sorted_diffs = np.sort(diffs)[::-1]
    sorted_indices = np.argsort(diffs)[::-1]

    return sorted_indices[:top_num]




@app.route('/', methods = ["GET", "POST"])
def index():
    #diveraging_indices = DivergingInstancesFinder(100).tolist()
    #hard code
    diveraging_indices = [669, 252, 120, 480, 628, 945, 542, 852, 984, 658, 183, 23, 930, 827, 579, 394, 635, 849, 958, 216, 476, 190, 271, 43, 445, 758, 788, 612, 103, 751, 737, 939, 320, 602, 807, 640, 192, 856, 898, 454, 381, 764, 418, 588, 465, 630, 52, 529, 108, 316, 743, 485, 753, 623, 554, 502, 894, 6, 805, 760, 312, 286, 217, 632, 382, 997, 985, 374, 750, 506, 409, 705, 379, 42, 332, 993, 361, 590, 855, 308, 627, 413, 269, 290, 735, 584, 483, 688, 557, 141, 663, 661, 664, 283, 971, 67, 995, 932, 107, 531]

    data = Loss_Difference_Summary(losses_instance_data,bin_counts,sort_index)
    #data = jsonifydata(loss_data)
    return render_template('index.html',data = data, num_of_nn = num_of_nn, num_of_epoch = num_of_epoch, classes = list(classes_pos_neg), tsne_data= tsne_data.tolist(), classes_n = list(classes), loss_diff_data =losses_instance_data.tolist(), predict_data = predictions_data.tolist(), indices= diveraging_indices)


@app.route('/class_data', methods =["GET", "POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_class_data():
    epoch_idx,class_name,modelid = json.loads(request.get_data())
    class_name = class_name[:-3]
    class_idx = classes.index(class_name)
    model_idx = int(modelid[5:])
    epoch_idx = int(epoch_idx)
    print('epoch:',epoch_idx,'class:',class_name,'class_idx:',class_idx,'model_idx:',model_idx)
    selected_index = sort_index[bin_counts[class_idx]:bin_counts[class_idx+1]]


    #shape of losses_instance_data: num_of_instances X number_of_nn X num_of_epoch

    #print(losses_instance_data.shape)

    #print(losses_instance_data[selected_index].shape)

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

    input_data = losses_instance_data[brushed_index]


    # overall_data = np.transpose(losses_instance_data[brushed_index],(1,2,0)) #shape: nn X epoch X instances

    sub_true_label = np.array(true_label)[brushed_index]

    label_counts = np.zeros(10)

    for label in sub_true_label:
        label_counts[label] +=1

    sort_true_label = np.sort(sub_true_label)
    sort_index = np.argsort(sub_true_label)

    sub_bin_counts = np.insert(np.cumsum(label_counts),0,0).astype(int)


    loss_data_sub = Loss_Difference_Summary(input_data,sub_bin_counts,sort_index)

    return jsonify(loss_data_sub)




@app.route('/instance_data', methods=["GET","POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_image_data():
    data_summary = []
    epoch_0_0,epoch_0_1,epoch_1_0,epoch_1_1,selectedDot = json.loads(request.get_data())
    epochs =[[int(epoch_0_0),int(epoch_0_1)],[int(epoch_1_0),int(epoch_1_1)]]

    print('epoch for model 1: ',epochs[0],"epoch for model 2: " ,epochs[1], "Index:", selectedDot)

    for i,nn_chosen in enumerate(num_of_nn):

        data_s  =[]

        epoch_chosen = epochs[i]

        data_file = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_chosen,epoch_chosen[1])))

        data_file_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_chosen,epoch_chosen[0])))


        grad_prev = utils.GradientBackPropogation(nn_chosen,epoch_chosen[0],selectedDot)['c1']

        grad = utils.GradientBackPropogation(nn_chosen,epoch_chosen[1],selectedDot)['c1']

        weight = np.mean(grad, axis=1)
        weight_prev = np.mean(grad_prev,axis=1)
        matrix_c1 = data_file[0]['c1'][selectedDot]
        matrix_c1_prev = data_file_prev[0]['c1'][selectedDot]

        gradcam = utils.GradCamAlgorithm(weight,matrix_c1,[28,28])

        gradcam_prev = utils.GradCamAlgorithm(weight_prev,matrix_c1_prev,[28,28])
        #gradcam_prev = np.zeros(784)

        for label in ['c1','c2','f1','o']:
            matrix = data_file[0][label][selectedDot]
            matrix_prev = data_file_prev[0][label][selectedDot]
            data_s.append({'label': label, 'data_origin': matrix.flatten().tolist(), 'data_prev': matrix_prev.flatten().tolist()})

        data_s.append({'label':'input','data_origin':gradcam.tolist(), 'data_prev': gradcam_prev.tolist()})

        data_summary.append(data_s)

    return jsonify(data_summary)

def SearchActiviationDiff(nn,epochs,layer_id,index_dot):
    data_file = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epochs[1])))[0][layer_id].reshape(num_of_input,-1)

    data_file_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epochs[0])))[0][layer_id].reshape(num_of_input,-1)

    data_diff = data_file - data_file_prev

    #print(data_diff.shape)

    data_distance = []

    for i in range(num_of_input):
        # distance = np.linalg.norm(data_diff[i]-data_diff[index_dot])/(np.linalg.norm(data_diff[index_dot]) * np.linalg.norm(data_diff[i]))
        distance = np.sqrt(2*(1.0- np.clip(np.inner(data_diff[i],data_diff[index_dot])/(np.linalg.norm(data_diff[i])*np.linalg.norm(data_diff[index_dot])),-1.0,1.0)))
        data_distance.append(float(distance))


    # toplist = np.argsort(data_distance)
    distancelist = data_distance
    topdata = data_diff
    return topdata.tolist(), distancelist


@app.route('/search_instance_data', methods=["GET","POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def searh_instance_data():
    top_num= 1000
    data_summary = []
    epochs =[]
    layer_id,[epochs_1,epochs_2],selectedDot = json.loads(request.get_data())
    epochs_1 = list(map(int, epochs_1))
    epochs_2 = list(map(int, epochs_2))
    print('epoch for model 1: ',epochs_1,"epoch for model 2: " ,epochs_2, "Index:", selectedDot,"Layer: ", layer_id)


    # NN_data_1,sort_indices_1,NN_distances_1 = SearchActiviationDiff(num_of_nn[0],epochs[0],layer_id,selectedDot)

    # NN_data_2,sort_indices_2,NN_distances_2 = SearchActiviationDiff(num_of_nn[1],epochs[1],layer_id,selectedDot)

    NN_data_1,NN_distances_1 = SearchActiviationDiff(num_of_nn[0],epochs_1,layer_id,selectedDot)
    NN_data_2,NN_distances_2 = SearchActiviationDiff(num_of_nn[1],epochs_2,layer_id,selectedDot)

    # data_points=set([])


    #TODO: I foridden the rank right now
    # s1=0
    # s2=0
    # while len(data_points) < top_num:
    #    # print(NN_distances_1[sort_indices_1[s1]],NN_distances_2[sort_indices_2[s2]])
    #     if NN_distances_1[sort_indices_1[s1]] <= NN_distances_2[sort_indices_2[s2]]:
    #         data_points.add(sort_indices_1[s1])
    #         s1= s1+1
    #     else:
    #         data_points.add(sort_indices_2[s2])
    #         s2 = s2+1

    #print(s1,s2)

    # data_points.remove(selectedDot)
    # data_points = list(data_points)

    #print(len(data_points))

    result=[]

    for i in range(top_num):
        # index = data_points[i]
        #print(type(index), type(true_label[index]), type(NN_distances_1[i]), type(NN_distances_2[i]))
        # data_point = {"label":true_label[index],"index": index, "x": NN_distances_1[index], "y": NN_distances_2[index], "v1": NN_data_1[index], "v2": NN_data_2[index]}
        data_point = {"label":true_label[i],"index": i, "x": NN_distances_1[i], "y": NN_distances_2[i], "v1": NN_data_1[i], "v2": NN_data_2[i]}
        #print(data_point)
        result.append(data_point)
    #print(result)

    return jsonify(result)


@app.route('/grad_instance_data', methods=["GET","POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def grad_data():
    modelID,epoch_chosen_prev,epoch_chosen,selectedDot,label_clicked= json.loads(request.get_data())
    model_idx = int(modelID[-1])
    epoch_idx = int(epoch_chosen)
    epoch_idx_prev = int(epoch_chosen_prev)
    selectedDot = int(selectedDot)
    #print(label_clicked)
    labels = [i  for i in range(10) if label_clicked[str(i)]==1]
    print('model: ',model_idx,'epoch:',epoch_idx,'epoch_prev:',epoch_idx_prev, "Index:", selectedDot,"Label:",labels)

    gradcam_diff=[]
    data_file = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(model_idx,epoch_idx)))

    data_file_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(model_idx,epoch_idx_prev)))

    grad_prev = None
    grad = None

    if labels ==[]:

        grad_prev = utils.GradientBackPropogation(model_idx,epoch_idx_prev,selectedDot)

        grad= utils.GradientBackPropogation(model_idx,epoch_idx,selectedDot)
    else:
        grad_prev = utils.GradientBackPropogation(model_idx,epoch_idx_prev,selectedDot,labels)

        grad = utils.GradientBackPropogation(model_idx,epoch_idx,selectedDot,labels)

    weight = np.mean(grad['c1'], axis=1)
    weight_prev = np.mean(grad_prev['c1'],axis=1)
    matrix_c1 = data_file[0]['c1'][selectedDot]
    matrix_c1_prev = data_file_prev[0]['c1'][selectedDot]

    gradcam = utils.GradCamAlgorithm(weight,matrix_c1,[28,28])

    gradcam_prev = utils.GradCamAlgorithm(weight_prev,matrix_c1_prev,[28,28])

    gradcam_diff.append({'label':'input','data':(gradcam - gradcam_prev).tolist()})
    #gradcam_diff.append({'label':'input','data':gradcam.tolist()})
    for label in ['c1','c2','f1']:

        w = np.mean(grad[label], axis=1)
        w_prev = np.mean(grad_prev[label],axis=1)
        m_c1 = data_file[0][label][selectedDot]
        m_c1_prev = data_file_prev[0][label][selectedDot]

        diff =  (w*(m_c1.reshape((m_c1.shape[0],-1)).T)).T - (w_prev*(m_c1_prev.reshape((m_c1_prev.shape[0],-1)).T)).T
        gradcam_diff.append({'label':label,'data':diff.flatten().tolist()})



    return jsonify(gradcam_diff)











if __name__ == '__main__':
    #print(true_label)
    app.run(host='0.0.0.0', port = 5000)


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



# @app.route('/data', methods =["GET", "POST"])
# @cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
# def get_data():
#     nn_idx,class_idx, epoch_idx = json.loads(request.get_data())
#     nn_idx= int(nn_idx)
#     class_idx=int(class_idx)
#     epoch_idx= int(epoch_idx)
#     print('nn_num:',nn_idx, 'class_idex:', class_idx, "epoch:" ,epoch_idx)
#     if(nn_idx !=0 and epoch_idx>1):
#         last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch_idx)))[0]['o']
#         last_layer_data_prev = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch_idx-1)))[0]['o']

#         selected_index = sort_index[bin_counts[class_idx]:bin_counts[class_idx+1]]

#         last_layer_diff = last_layer_data[selected_index] - last_layer_data_prev[selected_index]


#         tsne_result = utils.TSNE_Caculator(last_layer_diff)

#         loss_before = utils.SELoss_Caculator(last_layer_data_prev[selected_index],true_label[selected_index])


#         loss_after = utils.SELoss_Caculator(last_layer_data[selected_index],true_label[selected_index])

#         #print(loss_before,loss_after)

#         # corrects = np.argmax(last_layer_data[selected_index], axis=1).tolist()

#         # corrects_prev = np.argmax(last_layer_data_prev[selected_index], axis=1).tolist()
#         # last_layer_data_prev =

#         # true_labels = true_label[selected_index].tolist()
#         colorArray = utils.Loss2RGB(loss_before, loss_after)[0].tolist()


#         #print(colorArray)

#         points_summary = {'position': tsne_result, 'colorArray': colorArray, 'index': selected_index.tolist()}

#         # points_summary = {'position': tsne_result, 'correctness': corrects, 'correctness_prev': corrects_prev, 'index': selected_index.tolist(), 'true_label': true_labels}
#         return jsonify(points_summary)



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
# def Training_process_data():
#     data_matrices = []


#     for epoch in range(1, num_of_epoch+1):
#         accuracy_data = []
#         predict_label = []
#         for nn in range(1,num_of_nn+1):
#             last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[1]

#         #predict_label = np.load(os.path.join(data_path,'predict_label_epoch{}.npy'.format(epoch)))
#             predict_label.append(last_layer_data)

#         #shape:num_of_nn X num_of_input
#         predict_label =np.array(predict_label)
#         #print(predict_label.shape)
#         sort_predict_label = predict_label[:,sort_index]
#         #print(sort_predict_label.shape)

#         corrects = [ sort_predict_label[i]==sort_true_label for i in range(predict_label.shape[0]) ]
#         corrects = np.array(corrects)
#         #print(corrects)
#         corrects_classes = np.array([np.sum(corrects[:,bin_counts[i]:bin_counts[i+1]],axis =1 )/(bin_counts[i+1]-bin_counts[i]) for i in range(10)])

#         #print(corrects_classes.shape)


#         data_matrices.append(np.transpose(corrects_classes,(1,0)) )
#         #print(data_matrices)

#     return data_matrices


# def Loss_Data_Change():
#     data_matrices = []
#     for epoch in range(0, num_of_epoch+1):
#         accuracy_data = []
#         losses= []
#         for nn in range(1,num_of_nn+1):
#             last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[0]['o']

#         #predict_label = np.load(os.path.join(data_path,'predict_label_epoch{}.npy'.format(epoch)))
#             losses.append(last_layer_data)

#         #shape:num_of_nn X num_of_input X 10
#         losses = np.array(losses)
#         #print(losses.shape)
#         sort_losses = losses[:,sort_index]
#         #print(sort_losses.shape)

#         losses_sum = [utils.log_loss_Caculator(sort_losses[i],sort_true_label) for i in range(sort_losses.shape[0]) ]

#         losses_sum = np.array(losses_sum)
#         #print(losses_sum.shape)

#         loss_classes = np.array([np.sum(losses_sum[:,bin_counts[i]:bin_counts[i+1]],axis =1 ) for i in range(10)])

#         data_matrices.append(np.transpose(loss_classes,(1,0)) )

#     return data_matrices




# def TSNE_Caculator():
#     TSNE_data = []
#     for epoch in range(1, num_of_epoch+1):
#         TSNE_epoch_data = []
#         for nn in range(1,num_of_nn+1):
#             last_layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn,epoch)))[0]['o']
#             tsne = TSNE(n_components=2)
#             print("tNSE")
#             trans_data = tsne.fit_transform(last_layer_data)
#             print(trans_data.shape)
#             TSNE_epoch_data.append(trans_data.tolist())
#         TSNE_data.append(TSNE_epoch_data)

#     return TSNE_data
