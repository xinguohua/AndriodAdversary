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

from secondattack.fgsmattack.advbox.adversary import Adversary
from secondattack.fgsmattack.advbox.attacks.deepfool import DeepFoolAttack
from secondattack.fgsmattack.advbox.models.keras import KerasModel

import tensorflow as tf

import keras

#pip install keras==2.1

def main():


    # 设置为测试模式
    keras.backend.set_learning_phase(0)

    #加载模型
    model = tf.keras.models.load_model('..//..//..//malwareclassification//models//best_model_200_200.h5')

    #打印模型信息
    logging.info(model.summary())


    #读入一个样本数据
    data = np.loadtxt(open("..//..//..//data//onerow.csv","rb"), delimiter=",", skiprows=0, dtype=np.float32)
    data=np.matrix(data)
    print(data)


    #获取输出层
    # keras中获取指定层的方法为：
    # base_model.get_layer('block4_pool').output)
    logits=model.get_layer('dense_3').output


    # advbox demo
    # 因为原始数据没有归一化  所以bounds=(0, 255)  KerasMode内部在进行预测和计算梯度时会进行预处理
    # imagenet数据集归一化时 标准差为1  mean为[104, 116, 123]
    #初始化模型
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

    adversary = Adversary(data,None)

    tlabel = 0
    adversary.set_target(is_targeted_attack=True, target_label=tlabel)

    # deepfool targeted attack
    adversary,featurenumber,m,n = attack(adversary, **attack_config)

    # featurenumber为扰动个数
    print(featurenumber)
    if adversary.is_successful():
        # 得到对抗样本<class 'numpy.ndarray'>
        print(adversary.adversarial_example[0])
        print(
            'nonlinear attack success, adversarial_label=%d'
            % (adversary.adversarial_label) )





    print("second fgsm target attack done")



if __name__ == '__main__':

    main()
