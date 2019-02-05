## Questions users could answer using the visualization

1. At which epoch, the model has learned to distinguish one class from the others?
2. What is the sequence of learning different classes for different models? Is one class always learned faster than another?
3. How are the predictions of the selected instances changing at this epoch?
4. How are the losses of the selected instances changing at this epoch?
5. Which instances that have bad prediction at previous epoch before are   predicted correctly at this epoch?
6. Which instances that have small predictions loss at previous epoch are predicted incorrectly at this epoch?
7. How does the response of model to the instance(instances) change at this epoch?
8. What are the important regions of the instance image when the model is making prediction at this epoch?
9. What are the important neurons of the full connected layer or important regions of The convolutional layer when the model is making a prediction?
10. When an increase of a prediction accuracy occurs for specific class, are the same neurons being activated?


# Use Case
1. User would like to compare two models trained over the same dataset. The user could use the vis tool to explore the models and instances. Find the instances which behaves similarly of differently in both models and figure out the behaviors of the internal structure. For example, for an image, when the accuracy of prediction increase for both models, we could hypothesize that two models are observing the same part of the image and the corresponding activated neurons of the hidden layers are responsible for that part. In addition, for an instance which two models predicted differently, a user could also figure out why this happens in terms of which region or which neurons are the two models concentrated on by using the vis.

2. User could also focus on a specific model of interest and analyze the instances. For example, a user may be interested in the how the model distinguishes “5” and “8” in the MNIST dataset. He could select the epoch when “5” images have a huge accuracy increase and the epoch when “8” have a huge accuracy increase. Then he could select one “5” instance and one “8” instance and observe the response of the neural network to these two instance at two epochs. This could have the potentials to answer questions like which region or which neuron is important when the neural network is trying to learn “5” or “8”.  Or if I have another “5”, is it always activating the same region or neurons?

3. User could also use this vis to debug the training process. For example, a training process gets stuck, the model has low accuracy in “Truck” of the cifar10 dataset. We hypothesize that it may be caused by the fact that the models are not focusing on outline of the truck but it is looking at the background instead.  In order to verify that  we could choose the “Truck” instances at several epochs. by exploring the internal response to these instances of the model, we could know the important regions of images for this model.







