import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from joblib import load

import os




def load_models():
    """
    load saved models (classic ml & neural nets with different optimizers)
    """
    adam = tf.keras.models.load_model(path + "best_model_200_200.h5")
    sgd_mom = tf.keras.models.load_model(path + "best_model_200_100.h5")

    #classic ml
    rf = load(path+'model_RandomForestClassifier.joblib')
    lr = load(path+'model_LogisticRegression.joblib')

    return adam, sgd_mom, rf, lr


def final_prediction(adam, sgd, lr, rf):
    sum_pred = []
    for i in range(len(adam)):
        sum_pred.append([])
        for j in range(len(adam[i])):
            sum_pred[i].append((adam[i][j] + sgd[i][j]) + rf[i][j] + lr[i][j]/4)
    return sum_pred









if __name__ == "__main__":
    path = "..//malwareclassification//models//"
    total_features = 25000  # total unique features

    # 0，1型数据
    # 正常恶意样本预测,加载正常的恶意样本数据
    # 加载不同的攻击方法产生不同的对抗样本******
    # 加载JSMF的恶意软件和良性软件
    normal_data = np.loadtxt(open("..//data//jsmf//jsmf_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    normal_labels = np.loadtxt(open("..//data//jsmf//jsmf_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    begin_data = np.loadtxt(open("..//data//jsmf//jsmf_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    begin_labels = np.loadtxt(open("..//data//jsmf//jsmf_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # 加载onefeature的恶意软件和良性软件
    # normal_data = np.loadtxt(open("..//data//onefeature//onefeature_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # normal_labels = np.loadtxt(open("..//data//onefeature//onefeature_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # begin_data = np.loadtxt(open("..//data//onefeature//onefeature_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # begin_labels = np.loadtxt(open("..//data//onefeature//onefeature_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # 加载deepfool的恶意软件和良性软件
    # normal_data = np.loadtxt(open("..//data//deepfool//deepfool_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # normal_labels = np.loadtxt(open("..//data//deepfool//deepfool_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # begin_data = np.loadtxt(open("..//data//deepfool//deepfool_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # begin_labels = np.loadtxt(open("..//data//deepfool//deepfool_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # 加载fgsm的恶意软件和良性软件
    # normal_data = np.loadtxt(open("..//data//fgsm//fgsm_200_200_X_normal.csv", "rb"), delimiter=",", skiprows=0,
    #                          dtype=np.int32)
    # normal_labels = np.loadtxt(open("..//data//fgsm//fgsm_200_200_Y_normal.csv", "rb"), delimiter=",", skiprows=0,
    #                            dtype=np.int32)
    # begin_data = np.loadtxt(open("..//data//fgsm//fgsm_200_200_X_begin.csv", "rb"), delimiter=",", skiprows=0,
    #                         dtype=np.int32)
    # begin_labels = np.loadtxt(open("..//data//fgsm//fgsm_200_200_Y_begin.csv", "rb"), delimiter=",", skiprows=0,
    #                           dtype=np.int32)
    # 拼接矩阵
    # 将良性软件与正常恶意软件拼接
    val_data = np.concatenate((normal_data, begin_data), axis=0)
    val_labels = np.concatenate((normal_labels, begin_labels), axis=0)

    print("原始预测==============================")
    adam, sgd_mom, rf, lr = load_models()

    adam_pred = adam.predict(val_data)
    sgd_mom_pred = sgd_mom.predict(val_data)
    rf_pred = rf.predict_proba(val_data)
    lr_pred = lr.predict_proba(val_data)
    predictions = final_prediction(adam_pred, sgd_mom_pred, rf_pred, lr_pred)

    confusion = confusion_matrix(val_labels, np.argmax(predictions, axis=1))
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    FNR = FN / float(FN + TP) * 100
    FPR = FP / float(FP + TN) * 100
    accuracy = ((TP + TN) / float(TP + TN + FP + FN)) * 100
    print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
    print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

    print("对抗预测=================")
    # ================对抗样本+原始良性软件预测
    # 加载jsmf对抗样本
    adv_data = np.loadtxt(open("..//data//jsmf//jsmf_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    adv_labels = np.loadtxt(open("..//data//jsmf//jsmf_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # 加载onefeature对抗样本
    # adv_data = np.loadtxt(open("..//data//onefeature//onefeature_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # adv_labels = np.loadtxt(open("..//data//onefeature//onefeature_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # 加载deepfool对抗样本
    # adv_data = np.loadtxt(open("..//data//deepfool//deepfool_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)
    # adv_labels = np.loadtxt(open("..//data//deepfool//deepfool_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,dtype=np.int32)

    # 加载deepfool对抗样本
    adv_data = np.loadtxt(open("..//data//fgsm//fgsm_200_200_X_adv.csv", "rb"), delimiter=",", skiprows=0,
                          dtype=np.int32)
    adv_labels = np.loadtxt(open("..//data//fgsm//fgsm_200_200_Y_adv.csv", "rb"), delimiter=",", skiprows=0,
                            dtype=np.int32)
    # 拼接矩阵
    # 将良性软件与正常恶意软件拼接
    val_data1 = np.concatenate((adv_data, begin_data), axis=0)
    val_labels1 = np.concatenate((adv_labels, begin_labels), axis=0)

    # evaluate the models on adversarial examples
    adam, sgd_mom, rf, lr = load_models()
    adam_pred = adam.predict(val_data)
    sgd_mom_pred = sgd_mom.predict(val_data)
    rf_pred = rf.predict_proba(val_data)
    lr_pred = lr.predict_proba(val_data)

    predictions = final_prediction(adam_pred, sgd_mom_pred, rf_pred, lr_pred)

    confusion = confusion_matrix(val_labels, np.argmax(predictions, axis=1))
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    FNR = FN / float(FN + TP) * 100
    FPR = FP / float(FP + TN) * 100
    accuracy = ((TP + TN) / float(TP + TN + FP + FN)) * 100
    print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
    print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)