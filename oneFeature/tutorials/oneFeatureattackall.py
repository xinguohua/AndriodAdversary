#coding=utf-8

# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FGSM tutorial on mnist using advbox tool.
FGSM method is non-targeted attack while FGSMT is targeted attack.
"""
from __future__ import print_function
import sys
sys.path.append("..")
import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)

#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#pip install Pillow

from oneFeature.advbox.adversary import Adversary
from oneFeature.advbox.attacks.deepfool import DeepFoolAttack
from oneFeature.advbox.models.keras import KerasModel

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array,array_to_img
from keras.applications.resnet50 import decode_predictions
from oneFeature.sa import utilities as ut
import tensorflow as tf

import keras

#pip install keras==2.1

def main():


    # 设置为测试模式
    keras.backend.set_learning_phase(0)

    #加载模型
    model = tf.keras.models.load_model('..//..//malwareclassification//models//best_model_200_200.h5')

    #打印模型信息
    logging.info(model.summary())

    # 获取输出层
    # keras中获取指定层的方法为：
    # base_model.get_layer('block4_pool').output)
    logits = model.get_layer('dense_3').output

    # advbox demo
    # 因为原始数据没有归一化  所以bounds=(0, 255)  KerasMode内部在进行预测和计算梯度时会进行预处理
    # imagenet数据集归一化时 标准差为1  mean为[104, 116, 123]
    # 初始化模型
    m = KerasModel(
        model,
        model.input,
        None,
        logits,
        None,
        bounds=(0, 1),
        channel_axis=0,
        preprocess=None,
        featurefqueezing_bit_depth=1)

    attack = DeepFoolAttack(m)
    attack_config = {"iterations": 200, "overshoot": 10}
    tlabel = 0

    #读入所有样本数据
    val_data = np.loadtxt(open("..//..//data//x_test01.csv", "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    val_labels = np.loadtxt(open("..//..//data//y_test01.csv", "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    #data = np.loadtxt(open("D:\\data\\onerow.csv","rb"), delimiter=",", skiprows=0, dtype=np.float32)
    #测试集中恶意软件的数量
    malwarenumber=0
    #所有扰动的数量
    allchangefeaturenumber=0
    #存放所有对抗样本的list
    advmalware=[]

    # 画图
    allX, allY = [], []  # [[样本1迭代次数.....][样本2迭代次数....]....] [[bestvalues...][bestvalues...]]

    for i in range(len(val_data)):
    # 测试
    # for i in range(5):
        if val_labels[i] == 1:
            malwarenumber=malwarenumber+1
            data = val_data[i:i + 1]
            data=np.matrix(data)
            print("=========================="+str(i))
            print(data)

            adversary = Adversary(data, None)

            adversary.set_target(is_targeted_attack=True, target_label=tlabel)


            # deepfool targeted attack
            #对抗样本，改变特征的数量，迭代次数横坐标，bestvalues纵坐标
            adversary,featurenumber,x,y = attack(adversary, **attack_config)
            allchangefeaturenumber=allchangefeaturenumber+featurenumber

            # 增加特征x[....]==返回修改特征的迭代次数横坐标
            allX.append(x)
            # 增加特征x的[bestvalus...]bestvalues变化的纵坐标
            y = ut.normalization(y)
            # 进行归一化 因为量级不一样
            allY.append(y)

            if adversary.is_successful():
                advmalware.append(adversary.adversarial_example[0])
                print(
                    'nonlinear attack success, adversarial_label=%d'
                    % (adversary.adversarial_label))
            del adversary
            print("deepfool target attack done==========+"+str(i)+"个样本完成")


    #打印所有扰动的数量
    print("所有扰动的数量"+str(allchangefeaturenumber))
    #打印恶意软件的数量
    print("恶意软件的数量"+str(malwarenumber))
    #求平均扰动
    avechangefeaturenumber=allchangefeaturenumber/malwarenumber
    print("平均扰动"+str(avechangefeaturenumber))

    #针对某一架构DNN的对抗样本存到csv文件中
    advmalware=np.mat(advmalware)
    np.savetxt('..//..//data//onefeature_200_200.csv', advmalware, delimiter = ',')

    # 画图
    # 单个样本 过程
    # 每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（红线）
    save_name =  "AllMalwarefeaturesbestSA" + "_" + "_sa_x_fitnees_pic"
    # 迭代次数和fitness图
    ut.plotgraph(allX[0:30], allY[0:30],
                     "allmalwarefeaturesbestSA" + "_" + "_sa_x_fitnees_pic",
                    save_name)






if __name__ == '__main__':

    main()
