import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import os
import malwareclassification.neural_network as NN
import random
from sklearn.model_selection import train_test_split
import pandas as pd



"""
functions to compute Jacobian with numpy.
https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180
First we specify the the forward and backward passes of each layer to implement backpropagation manually.
函数以numpy计算Jacobian。
https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180
首先，我们指定每层的正向和反向传递，以手动实现反向传播。
"""





def adversarial_training():
    # model, epochs, batch_size, features, labels, verbose = 0
    NN.train_neural_network(trained_model, 4, 15, val_data, val_labels, verbose=2)
    # trained_model.save('adversarial_jsmf_model.h5')
    trained_model.save('adversarial_fgsm_model.h5')
    predictions = trained_model.predict(val_data)
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


if __name__ == "__main__":
    total_features = 25000  # total unique features

    data = pd.read_csv('..//..//data//x_train01.csv')
    labels = pd.read_csv('..//..//data//y_train01.csv')
    # 将训练数据拆分成0.8训练 0.2 验证
    data, test_data, labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                            random_state=0)  # 随机选择25%作为测试集，剩余作为训练集




    #加载200_200架构的模型
    trained_model = tf.keras.models.load_model('..//..//malwareclassification//models//best_model_200_200.h5')

    #加载混合的对抗样本
    ##两种 一种fgsm,一种jsma
    # 把两种对抗样本混合起来
    #读取对抗样本jsmf(fgsm)
    # advdata = np.loadtxt(open("..//..//data//adversarytrain//jsmf_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0, dtype=np.float32)
    # advdlabel = np.loadtxt(open("..//..//data//adversarytrain//jsmf_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.float32)
    advdata = np.loadtxt(open("..//..//data//adversarytrain//fgsm_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,
                         dtype=np.float32)
    advdlabel = np.loadtxt(open("..//..//data//adversarytrain//fgsm_200_200_Y_adv.csv", "rb"), delimiter=",",
                           skiprows=0, dtype=np.float32)
    advdata = np.matrix(advdata)
    advdlabel = np.matrix(advdlabel).T
    #合并对抗样本
    val_data = np.concatenate((data, advdata), axis=0)
    val_labels = np.concatenate((labels, advdlabel), axis=0)
    print("data",data.shape)
    print("advdata",advdata.shape)
    print("label",labels.shape)
    print("advlabel",advdlabel.shape)
    print("val_data",val_data.shape)
    print("val_lables",val_labels.shape)
    #使用val_data,val_labels训练
    adversarial_training()
