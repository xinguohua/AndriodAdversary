import numpy as np

import pandas as pd

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


#读取csv
#读取对抗样本
# adv_data = pd.read_csv('./featuresfinal/x_adv_combine.csv',usecols=['combine'])  # 读取训练数据
adv_data = pd.read_csv('./featuresfinal/x_adv_aleatoric.csv',usecols=['aleatoric'])
# adv_data = pd.read_csv('./featuresfinal/x_adv_epistemic.csv',usecols=['epistemic'])
print("adv_data",adv_data.shape)
adv_data = np.array(adv_data)[:3000,:].reshape(1,3000)#np.ndarray()
adv_data = adv_data[0]
print(adv_data)


#读取正常
# normal_data = pd.read_csv('./featuresfinal/x_normal_combine.csv',usecols=['combine'])  # 读取训练数据
normal_data = pd.read_csv('./featuresfinal/x_normal_aleatoric.csv',usecols=['aleatoric'])
# normal_data = pd.read_csv('./featuresfinal/x_normal_epistemic.csv',usecols=['epistemic'])
print("normal_data",normal_data.shape)
normal_data = np.array(normal_data)[:3000,:].reshape(1,3000)#np.ndarray()
normal_data = normal_data[0]
print(normal_data)

densities_pos = adv_data #adv是阳性
densities_neg = normal_data #normal 是阴性
artifacts, labels = merge_and_generate_labels(densities_pos, densities_neg)

# # 合起来合成两列 一列数据 一列标签
file_name = 'bua_jsmf.npy'
# file_name = 'bue_jsmf.npy'
# file_name = 'buc_jsmf.npy'
data = np.concatenate((artifacts, labels), axis=1)
print(data.shape)
np.save(file_name, data)