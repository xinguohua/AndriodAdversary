import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.metrics import confusion_matrix
from malwareclassification import set_onehot_encoding as onehot
import os
import joblib
from malwareclassification import models





"""
functions to compute Jacobian with numpy.
https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180
First we specify the the forward and backward passes of each layer to implement backpropagation manually.
"""


def affine_forward(x, w, b):
    """
    Forward pass of an affine layer
    :param x: input of dimension (I, )
    :param w: weights matrix of dimension (I, O)
    :param b: biais vector of dimension (O, )
    :return output of dimension (O, ), and cache needed for backprop
    """
    out = np.dot(x, w) + b
    cache = (x, w)
    return out, cache


def affine_backward(dout, cache):
    """
    Backward pass for an affine layer.
    :param dout: Upstream Jacobian, of shape (M, O)
    :param cache: Tuple of:
      - x: Input data, of shape (I, )
      - w: Weights, of shape (I, O)
    :return the jacobian matrix containing derivatives of the M neural network outputs with respect to
            this layer's inputs, evaluated at x, of shape (M, I)
    """
    x, w = cache
    dx = np.dot(dout, w.T)
    return dx


def relu_forward(x):
    """ Forward ReLU
    """
    out = np.maximum(np.zeros(x.shape), x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Backward pass of ReLU
    :param dout: Upstream Jacobian
    :param cache: the cached input for this layer
    :return: the jacobian matrix containing derivatives of the M neural network outputs with respect to
             this layer's inputs, evaluated at x.
    """
    x = cache
    dx = dout * np.where(x > 0, np.ones(x.shape), np.zeros(x.shape))
    return dx


def softmax_forward(x):
    """ Forward softmax
    """
    exps = np.exp(x - np.max(x))
    s = exps / exps.sum()
    return s, s


def softmax_backward(dout, cache):
    """
    Backward pass for softmax
    :param dout: Upstream Jacobian
    :param cache: contains the cache (in this case the output) for this layer
    """
    s = cache
    ds = np.diag(s) - np.outer(s, s.T)
    dx = np.dot(dout, ds)
    return dx


def get_activations(model, layer_id, X):
    """
    Computes outputs of intermediate layers
    :param model: the trained model
    :param layer_id: the id of the layer that we want the output from
    :param X: input feature vector
    :return: output of layer (layer_id)
    """
    intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                  outputs=model.layers[layer_id].output)
    intermediate_output = intermediate_layer_model.predict(X)
    return intermediate_output


def forward_backward(model, x):
    """
    computes the forward derivative for the given input
    :param model: the trained model
    :param x: input feature vector
    :return: prediction result and forward derivative
    """
    layer_to_cache = dict()  # for each layer, we store the cache needed for backward pass
    forward_values = []

    for i in range(0, len(model.layers), 2):
        values = {}
        w, b = model.layers[i].get_weights()
        values['w'] = w
        values['b'] = b
        forward_values.append(values)

    # Forward pass
    a1, cache_a1 = affine_forward(x, forward_values[0]['w'], forward_values[0]['b'])
    _, cache_r1 = relu_forward(a1)
    r1 = get_activations(model, 0, x)
    forward_values[0]['a'] = a1
    forward_values[0]['cache_a'] = cache_a1
    forward_values[0]['r'] = r1
    forward_values[0]['cache_r'] = cache_r1

    for i, layer_index in zip(range(1, len(forward_values) - 1), range(2, len(model.layers), 2)):
        a, cache_a = affine_forward(forward_values[i - 1]['r'], forward_values[i]['w'], forward_values[i]['b'])
        _, cache_r = relu_forward(a)
        r = get_activations(model, layer_index, x)
        forward_values[i]['a'] = a
        forward_values[i]['cache_a'] = cache_a
        forward_values[i]['r'] = r
        forward_values[i]['cache_r'] = cache_r

    a, cache_a = affine_forward(forward_values[len(forward_values) - 2]['r'],
                                forward_values[len(forward_values) - 1]['w'],
                                forward_values[len(forward_values) - 1]['b'])
    forward_values[len(forward_values) - 1]['a'] = a
    forward_values[len(forward_values) - 1]['cache_a'] = cache_a
    out, cache_out = softmax_forward(a)

    # backward pass
    dout = np.diag(np.ones(out.size, ))  # the derivatives of each output w.r.t. each output.
    dout = softmax_backward(dout, cache_out)
    dout = affine_backward(dout, forward_values[len(forward_values) - 1]['cache_a'])

    for i in range(len(forward_values) - 2, 0, -1):
        dout = relu_backward(dout, forward_values[i]['cache_r'])
        dout = affine_backward(dout, forward_values[i]['cache_a'])

    dout = relu_backward(dout, forward_values[0]['cache_r'])
    dx = affine_backward(dout, forward_values[0]['cache_a'])

    return out, dx


def craft_adversarial_samples(x, y, F, k):
    """
    JSMA variant for adversarial examples crafting algorithm as described in https://arxiv.org/abs/1606.04435
    JSMA iteratively selects the most useful features to perturb a small magnitude of value until the target class is
    achived. The perturbed featured are selected based on the saliency map. Saliency maps are used for a network's
    visualization and describe which features are the most important for a particular output class. The goal
    is to eliminate those attributes from a legitimate sample and bring up the most important ones for the target class
    in oder to cause the model to misclassify. This is done by pushing the features away from the original label
    and closer to the target class.
    Steps:
    1)Compute the gradient of F with respect to the input X to estimate the direction in which a perturbation in X
      would change F's output. That is, compute the forward derivative (the Jacobian of the learned function for
      a legitimate sample).
                            ∇F(x)=(∂F(x))/∂x=[(∂F_j (x))/(∂x_i )]_(iϵ1…M,jϵ1…N)
      where x is the model’s input, F is the network, F(x) the predicted class, M the input dimension,
      N the output dimension, (i, j) is the derivative class of class j with respect to the input feature i.
      In essence, it computes the gradient of F with respect to input x to estimate the direction in which
      a perturbation in x would change the output. In backpropagation, the forward derivative is calculated
      with respect to the loss function and the gradients with respect to the network parameters with the goal of
      updating the weights. On the contrary, in JSMA the forward derivative is taken with respect to the network
      directly and the gradients with respect to the input data.
    2)Choose a perturbation δ of X with maximal positive gradient into the the target class y'.
      In other words, choose the index that maximizes the change into the target class 0 by changing X_i.
      The limitation is that we can only add features and not discard them as in a real world scenario an adversary doesnt want
      to 'break' the functionality of an application.
    Algorithm:
       Input x, y, F, k, I
       x_adv <- x
       Gamma = {1...|x|}
       while arg max_jF_j(x_adv) != y and ||δ_X|| < k do
           Compute the forward derivative ∇F(adv_x)
           i_max = arg max_j∈Γ∩I,X_j=0  ∂Fy(X)/∂Xj
           if i_max <= 0 then
              :return Failure
           end if
           adv_x_i_max = 1
           δ_x <- x_adv - x
           :return adv_x
    :param x: input feature vector
    :param y: target class
    :param F: the trained model
    :param k: index of the hidden layer
    :return: adversarial sample based on feature vector x
    """
    x_adv = x
    gamma = [1] * len(x)
    delta_x = [0]
    changes = 0

    # 返回概率值最大的索引 0是良性 1是恶性 如果恶意软件 0概率最大直接返回
    if np.argmax(F.predict(x_adv), 1) == 0:  # if misclassification achieved return adv_x
        return x_adv, -1

    #循环条件 类别没有干扰变成0 良性 and   delta_x 1范数<k=20  changes<20
    while np.argmax(F.predict(x_adv), 1) != y and np.linalg.norm(delta_x, ord=1) < k and changes < 20:
        # compute forward derivative (Jacobian)
        prob, forward_derivative = forward_backward(F, x_adv)

        tmp = np.multiply(forward_derivative[0], gamma)
        for i, feature in enumerate(x_adv[0]):
            #if feature == 1:
            if feature != 0:
                tmp[i] = 0
        i_max = np.argmax(tmp)
        if i_max <= 0:
            raise ValueError('FAILURE: We can only add features to an application!')

        x_adv[0][i_max] = 1
        delta_x = np.subtract(x_adv, x)
        # print(i_max)
        if i_max not in changes_dict:
            changes_dict[i_max] = 1
        else:
            changes_dict[i_max] += 1
        changes += 1
    print("Changes:", changes)

    return x_adv, changes


def evaluate_other_models():
    """
    specify models to evaluate adv examples produced from another model. We give two examples with a neural net with
    SGD optimizer and complement naive bayes.
    """
    second_trained_model = tf.keras.models.load_model('model_SGD.h5')
    predictions = second_trained_model.predict(val_data)
    confusion = confusion_matrix(val_labels, np.argmax(predictions, axis=1))
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    FNR = FN / float(FN + TP) * 100
    FPR = FP / float(FP + TN) * 100
    accuracy = ((TP + TN) / float(TP + TN + FP + FN)) * 100
    print("Adversarial  FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
    print("Adversarial Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

    '''CNB = models.ComplementNaiveBayes()
    second_trained_model = load("models_incremental_learning/model_incremental_ComplementNaiveBayes.joblib")
    CNB.test_complement_naive_bayes_classifier(second_trained_model, val_data, val_labels)'''

def craft_deepfool_samples(x, y, F, k):

    x_adv = x
    gamma = [1] * len(x)
    delta_x = [0]
    changes = 0

    # 返回概率值最大的索引 0是良性 1是恶性 如果恶意软件 0概率最大直接返回
    if np.argmax(F.predict(x_adv), 1) == 0:  # if misclassification achieved return adv_x
        return x_adv, -1



    #循环条件 类别没有干扰变成0 良性 and   delta_x 1范数<k=20  changes<20
    while np.argmax(F.predict(x_adv), 1) != y and np.linalg.norm(delta_x, ord=1) < k and changes < 20:
        # compute forward derivative (Jacobian)
        prob, forward_derivative = forward_backward(F, x_adv)

        tmp = np.multiply(forward_derivative[0], gamma)
        for i, feature in enumerate(x_adv[0]):
            #if feature == 1:
            if feature != 0:
                tmp[i] = 0
        i_max = np.argmax(tmp)
        if i_max <= 0:
            raise ValueError('FAILURE: We can only add features to an application!')

        x_adv[0][i_max] = 1
        delta_x = np.subtract(x_adv, x)
        # print(i_max)
        if i_max not in changes_dict:
            changes_dict[i_max] = 1
        else:
            changes_dict[i_max] += 1
        changes += 1
    print("Changes:", changes)

    return x_adv, changes
if __name__ == "__main__":
    changes_dict = {}  # dictionary for perturbations (added features)
    trained_model = tf.keras.models.load_model('D:\\安全课程\\android detection\\Android-Malware-Detection-Adversarial-Examples-master\\malwareclassification\\best_model_Adam.h5')


    val_data = np.loadtxt(open("D:\\安全课程\\android detection\\Android-Malware-Detection-Adversarial-Examples-master\\drebin\\x_test.csv", "rb"), delimiter=",", skiprows=0,dtype=np.float32)
    val_labels = np.loadtxt(open("D:\\安全课程\\android detection\\Android-Malware-Detection-Adversarial-Examples-master\\drebin\\y_test.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)

    if val_labels[1] == 1:
            x = val_data[1:2]
            #print("x: ", x)
            #print(x.shape)
            try:
                adv_x, changes = craft_adversarial_samples(x, 0, trained_model, 20)
                # print(adv_x)
                print(changes)
            except NameError:
                pass
            except ValueError:
                pass

