from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from keras.models import load_model

from defence.detector.util import (get_data, get_noisy_samples, get_mc_predictions,
                      get_deep_representations, score_samples, normalize,
                      get_lids_random_batch, get_kmeans_random_batch)

# In the original paper, the author used optimal KDE bandwidths dataset-wise
#  that were determined from CV tuning
BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00}

# Here we further tune bandwidth for each of the 10 classes in mnist, cifar and svhn
# Run tune_kernal_density.py to get the following settings.
# BANDWIDTHS = {'mnist': [0.2637, 0.1274, 0.2637, 0.2637, 0.2637, 0.2637, 0.2637, 0.2069, 0.3360, 0.2637],
#               'cifar': [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#               'svhn': [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1274, 0.1000, 0.1000]}

#测试时用的路径
# PATH_DATA = "data/"
#正式的路径
PATH_DATA = "..//..//data//"


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


def get_kd(model, X_train, Y_train, X_test,X_test_adv):
    """
    Get kernel density scores
    :param model: 
    :param X_train:  训练样本---其实是测试样本
    :param Y_train:
    :param X_test: X_normal 正常的恶意样本
    :param X_test_adv:  X_adv 对抗的恶意样本
    :return: artifacts: positive and negative examples with kd values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    # Get deep feature representations
    print('Getting deep feature representations...') #得到神经网络最深层输出 X样本个数*200维(最后一个隐藏层的输出)
    X_train_features = get_deep_representations(model, X_train,
                                                batch_size=args.batch_size)
    X_test_normal_features = get_deep_representations(model, X_test,
                                                      batch_size=args.batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                   batch_size=args.batch_size)
    # Train one KDE per class 每个类训练一个kdes函数
    print('Training KDEs...')
    class_inds = {}
    for i in range(2): ##一个两个类别  遍历两个类别 Y_train中标签为i的所有索引
        class_inds[i] = np.where(Y_train == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")

    # 重新设置bandwidth********
    for i in range(2): #对每个类别循环 kd函数去拟合X_train_features(深层网络的输出)
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=3.7926) \
            .fit(X_train_features[class_inds[i]])
    # Get model predictions # Get model predictions 计算模型的预测
    print('Computing model predictions...')
    #计算X_test，X_test_adv预测类别 (为估计kds调用对应类别得密度函数)
    preds_test_normal = model.predict_classes(X_test, verbose=0,
                                              batch_size=args.batch_size)
    preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
                                           batch_size=args.batch_size)
    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    ) #X_test 得分
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    ) # X_adv 得分

    print("densities_normal:", densities_normal.shape)
    print("densities_adv:", densities_adv.shape)

    densities_pos = densities_adv #adv是阳性
    densities_neg = densities_normal #normal 是阴性
    artifacts, labels = merge_and_generate_labels(densities_pos, densities_neg)
    # 合起来
    return artifacts, labels


def get_lid(model, X_test,X_test_adv, k=10, batch_size=100):
    """
    Get local intrinsic dimensionality
    :param model: 模型
    :param X_test: 正常样本
    :param X_test_adv: 对抗恶意样本
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_adv = get_lids_random_batch(model, X_test, X_test_adv, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_adv:", lids_adv.shape)

    lids_pos = lids_adv
    lids_neg = lids_normal
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels


def main(args):
    assert args.attack in ['jsmf', 'deepfool', 'onefeature', 'fgsm'], \
        "Attack parameter must be either 'jsmf', 'deepfool', 'onefeature', 'fgsm'"
    #提取的特征
    assert args.characteristic in ['kd','lid'], \
        "Characteristic(s) to use 'kd', 'lid'"   #选择提取的特征

    #加载模型文件（只针对200_200架构）
    model_file = "..//..//malwareclassification//models//best_model_200_200.h5"

    print('Loading model...')

    # Load the model 加载模型
    model = load_model(model_file)



    # 大数据
    # 训练样本 在服务器上换成大数据
    #其实是测试数据
    X_train = np.loadtxt(
        PATH_DATA+"x_test01.csv",
        delimiter=",", skiprows=0, dtype=np.float32)
    Y_train = np.loadtxt(
        PATH_DATA+"y_test01.csv",
        delimiter=",", skiprows=0, dtype=np.float32)
    # 正常的恶意样本 到时候换成大数据 尺寸X_normal与X_adv一样
    X_normal = np.loadtxt(
        PATH_DATA+args.attack+'//'+args.attack+'_200_200_X_normal.csv',
        delimiter=",", skiprows=0, dtype=np.float32)
    Y_normal = np.loadtxt(
        PATH_DATA+args.attack+'//'+args.attack+'_200_200_Y_normal.csv',
        delimiter=",", skiprows=0, dtype=np.float32)
    # 对抗的恶意样本 到时候换成大数据
    X_adv = np.loadtxt(
        PATH_DATA+args.attack+'//'+args.attack+'_200_200_X_adv.csv',
        delimiter=",", skiprows=0, dtype=np.float32)
    Y_adv = np.loadtxt(
        PATH_DATA+args.attack+'//'+args.attack+'_200_200_Y_adv.csv',
        delimiter=",", skiprows=0, dtype=np.float32)
    print('Loading the data ...')

    #暂时不用 怕造成两者kd,lid与不确定性的长度不一样
    # #优化正常恶意软件X_normal，对抗性集，以仅包含模型正确分类原始版本的恶意软件样本
    # ################得到X_normal样本预测正确的索引
    # inds_correct=[]
    # print(X_normal)
    # for i in range(len(X_normal)):
    #     x_normal=X_normal[i:i+1]
    #     #pridct=np.argmax(model.predict(x_normal))
    #     #print(pridct)
    #     if np.argmax(model.predict(x_normal), 1)==Y_normal[i]:
    #         inds_correct.append(i) #得到所有预测正确恶意样本的索引
    # print("Number of correctly predict malware: %s" % (len(inds_correct)))
    #
    # #重新修正X_normal
    # X_normal = X_normal[inds_correct]
    # X_adv = X_adv[inds_correct]
    # #修正过后的形状
    # print("X_normal: ", X_normal.shape)
    # print("X_adv: ", X_adv.shape)

    if args.characteristic == 'kd':
        # extract kernel density
        characteristics, labels = get_kd(model, X_train, Y_train, X_normal,X_adv)
        print("KD: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        #要改############
        bandwidth =3.7926
        #攻击换成参数
        file_name = os.path.join(PATH_DATA, 'characteristic//%s_%s.npy'  % (args.characteristic, args.attack))
        # 合起来合成两列 一列数据 一列标签
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'lid':
        # extract local intrinsic dimensionality 提取lid
        characteristics, labels = get_lid(model, X_normal,X_adv,args.k_nearest, args.batch_size)
        print("LID: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        file_name = os.path.join(PATH_DATA, 'characteristic//%s_%s.npy'  % (args.characteristic, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #选择对应的攻击（样本）提取特征
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'jsmf', 'deepfool', 'onefeature', 'fgsm' ",
        required=True, type=str
    )
    #从(对抗样本)提取的特征
    parser.add_argument(
        '-r', '--characteristic',
        help="Characteristic(s) to use 'kd','lid'",
        required=True, type=str
    )
    #lid的参数
    parser.add_argument(
        '-k', '--k_nearest',
        help="The number of nearest neighbours to use; either 10, 20, 100 ",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=100)
    parser.set_defaults(k_nearest=5)
    args = parser.parse_args()
    main(args)
