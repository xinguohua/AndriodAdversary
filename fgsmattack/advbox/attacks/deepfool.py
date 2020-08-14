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
This module provide the attack method for deepfool. Deepfool is a simple and
accurate adversarial attack.
"""
from __future__ import division

from builtins import str
from builtins import range
import logging
import time
import numpy as np

from .base import Attack
from fgsmattack.sa.main import Main
import tensorflow as tf
from fgsmattack.sa import utilities as ut
__all__ = ['DeepFoolAttack']

def create_adversarial_pattern(input_x, input_y):
    """
    FGSM attack as described in https://arxiv.org/pdf/1412.6572.pdf
    The goal of FGSM is to cause the loss function to increase for specific inputs.
    It operates by perturbating each feature of an input x by a small value to maximize the loss.
    Steps:
    1)Compute the gradient of the loss with respect to the input
                            ∇_x J(θ,x,y)
      where x is the model's input, y the target class, θ the model's parameters, ∇_x the gradient and J(θ,x,y) the loss
    2)Take the sign of the gradient (calculated in 1), multiply it by a threshold ε and add it to the
       original input x.
                            x_adv=x+e*sign(∇_x J(θ,x,y))

    :param input_x: the original input data
    :param input_y: the original input label
    :return: the sign of the gradient
    """
    trained_model = tf.keras.models.load_model(
        '..//..//malwareclassification//models//best_model_200_200.h5')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_x)
        prediction = trained_model(input_x)  # predict original input
        loss = loss_object(input_y, prediction)  # get the loss
    # get the gradients of the loss with respect to the inputs
    gradient = tape.gradient(loss, input_x)
    # get the sign of the gradients to create perturbations
    signed_grad = tf.sign(gradient)
    return signed_grad


class DeepFoolAttack(Attack):
    """
    DeepFool: a simple and accurate method to fool deep neural networks",
    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
    https://arxiv.org/abs/1511.04599
    """

    def _apply(self, adversary, iterations=100, overshoot=0.02):
        """
          Apply the deep fool attack.

          Args:
              adversary(Adversary): The Adversary object.
              iterations(int): The iterations.
              overshoot(float): We add (1+overshoot)*pert every iteration.
          Return:
              adversary: The Adversary object.
          """
        assert adversary is not None

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()

        logging.info('min_={0}, max_={1}'.format(min_,max_))

        f = self.model.predict(adversary.original)

        if adversary.is_targeted_attack:
            labels = [adversary.target_label]
        else:
            max_class_count = 10
            class_count = self.model.num_classes()
            if class_count > max_class_count:
                #labels = np.argsort(f)[-(max_class_count + 1):-1]
                logging.info('selec top-{0} class'.format(max_class_count))
                labels= np.argsort(f)[::-1]
                labels = labels[0][:max_class_count]
            else:
                labels = np.arange(class_count)

        t= [ str(i) for i in labels]

        logging.info('select label:'+" ".join(t))

        gradient = self.model.gradient(adversary.original, pre_label)
        x = np.copy(adversary.original)

        for iteration in range(iterations):
            w = np.inf
            w_norm = np.inf
            pert = np.inf
            for k in labels:
                if k == pre_label:
                    continue
                #标签k对应的梯度
                gradient_k = self.model.gradient(x, k)
                #标签k对应的梯度和现有便签的梯度的差
                w_k = gradient_k - gradient
                #标签k和现有标签在logit层面的差距
                f_k = f[0][k] - f[0][pre_label]
                #保证非0 避免计算时除以0 梯度差距除以l2范式值
                w_k_norm = np.linalg.norm(w_k.flatten()) + 1e-8
                #标签k和现有标签在logit层面的差距除以l2范式的值
                pert_k = (np.abs(f_k) + 1e-8) / w_k_norm

                #对应论文中的选择最小的l
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                    w_norm = w_k_norm
                    #logging.info("k={0} pert={1} w_norm={2}".format(k,pert, w_norm))

            #论文中为+  advbox老版本实现成了- l2实现
            r_i = w * pert / w_norm

            #logging.info(r_i)

            # 放大系数 在原论文上的创新 提高攻击速度 论文中相当于overshoot=0
            x = x + (1 + overshoot) *r_i

            x = np.clip(x, min_, max_)

            f = self.model.predict(x)
            gradient = self.model.gradient(x, pre_label)
            adv_label = np.argmax(f)
            logging.info('iteration={}, f[pre_label]={}, f[target_label]={}'
                         ', f[adv_label]={}, pre_label={}, adv_label={}'
                         ''.format(iteration, f[0][pre_label], (
                             f[0][adversary.target_label]
                             if adversary.is_targeted_attack else 'NaN'), f[0][
                                 adv_label], pre_label, adv_label))


            if adv_label==0:
                print("linear attack success")
                break

        #deepfool 没成功
        if adv_label == 1:
            print("linear attack failed")
            #return adversary




        #deep fool迭代次数结束 开始离散化
        print("对抗样本deepfool线性解" + str(x))
        print("原始解" + str(adversary.original))
        # 在求模拟退火时输入必须是array
        x = np.array(x)
        m = Main(dataset=x[0], origin=adversary.original[0],model=self.model)
        #第一次迭代原始样本
        y=adversary.original[0].tolist()
        allX, allY = [], []  # [[特征1.....][特征2....]....] [[fitness...][fitness...]]
        valuesx, valuesy = [], []
        for iteration in range(iterations):

            fgsm_x = np.mat(y)
            fgsm_y = np.mat(pre_label)
            fgsm_x = tf.convert_to_tensor(fgsm_x, dtype=np.float32)
            fgsm_y = tf.convert_to_tensor(fgsm_y, dtype=np.int32)
            #扰动方向 有+1，有-1
            perturbations = create_adversarial_pattern(fgsm_x, fgsm_y)  # get the sign of gradient wrt the input
            print(perturbations)
            # 选出所有为1的索引,也就是扰动正方向的索引
            perturbationsindex = np.where(perturbations[0] == 1)[0]
            print(type(perturbationsindex))
            print(perturbationsindex)

            # 返回离散解至线性解的距离 离散解
            v, solutions,x,y = m.Run(y,iteration,perturbationsindex) #修改iteration特征，特征内进行模拟退火

            # 增加特征x[....]==返回best对应的那条曲线的横坐标
            allX.append(x)
            # 增加特征x的[fitness...]==返回best对应的那条曲线的纵坐标
            y = ut.normalization(y)
            # 进行归一化 因为量级不一样
            allY.append(y)
            # 特征坐标
            valuesx.append(iteration)
            # 特征对应的bestvalue
            valuesy.append(v)

            # 得到此增加一个特征下的离散解
            y=solutions.copy()
            maty = np.mat(solutions)
            print(maty)
            f = self.model.predict(maty)

            adv_label = np.argmax(f)
            logging.info('iteration={}, f[pre_label]={}, f[target_label]={}'
                         ', f[adv_label]={}, pre_label={}, adv_label={}'
                         ''.format(iteration, f[0][pre_label], (
                f[0][adversary.target_label]
                if adversary.is_targeted_attack else 'NaN'), f[0][
                                       adv_label], pre_label, adv_label))
            if adversary.try_accept_the_example(maty, adv_label):
                # 画图
                # 单个样本 过程
                # 每次特征取最好best那条曲线(修改几个特征几条曲线) 所有特征取平均（红线）
                save_name = str(iteration + 1) + "fgsmfeaturesbestSA" + "_" + "process" + str(int(time.time()))
                # 迭代次数和fitness图
                ut.plotavegraph(allX, allY,
                                str(iteration + 1) + "featuresbestSA" + "_" + "process",
                                save_name)

                # 画图
                # 单个样本 结果
                # bestvalue的曲线（每次修改特征的bestvalue）
                save_name1 = str(iteration + 1) + "fgsmfeaturesbestvalues" + "_" + "_results_pic" + str(int(time.time()))
                ut.plotsinglegraph(valuesx, valuesy,
                                   str(iteration + 1) + "featuresbestvalues" + "_" + "_results_pic",
                                   save_name1)
                return adversary,iteration+1,valuesx,valuesy



        return adversary
