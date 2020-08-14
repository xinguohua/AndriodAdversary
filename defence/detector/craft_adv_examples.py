from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.models import load_model
import pandas as pd
from jsma import craft_adversarial_samples



PATH_DATA = "data/"

def craft_one_type(model,X, Y, attack):
    """
    TODO
    :param model:
    :param X:
    :param Y:
    :param attack:
    :param batch_size:
    :return:
    """
    #恶意正常
    X_normal=[]
    Y_normal=[]
    #恶意对抗的
    X_adv=[]
    Y_adv =[]
    if attack == 'jsmf':
        # JSMF attack
        print('Crafting jsma adversarial samples. ')
        #for i in range(len(X)):
        for i in range(10):
            if Y[i] == 1:
                x = X[i:i + 1]
                X_normal.append(x[0])
                Y_normal.append(1)
                try:
                    adv_x, changes = craft_adversarial_samples(x, 0, model, 1)
                    # print(adv_x)
                    X_adv.append(adv_x[0])
                    Y_adv.append(1)

                except NameError:
                    pass
                except ValueError:
                    pass
    else:
        #one-pixel
        pass
    return X_normal,Y_normal,X_adv,Y_adv



def main(args):
    #选择攻击
    assert args.attack in [ 'jsmf', 'one-pixel'], \
        "Attack parameter must be either  " \
        "'jsma' or 'one-pixel' for attacking  detector"
    print('Attack: %s' % (args.attack))

    #模型文件
    model_file = "D:\\安全课程\\android detection\\zhenghe\\lid_adversarial_subspace_detection-master\\data\\best_model_Adam.h5"
    assert os.path.isfile(model_file), \
        'model file not found... must first train model using train_model.py.'
    #加载模型文件
    model = load_model(model_file)

    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)

    #加载测试集
    X_test = np.loadtxt(open("D:\\data\\x_test01.csv", "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    Y_test = np.loadtxt(open("D:\\data\\y_test01.csv", "rb"), delimiter=",", skiprows=0, dtype=np.int32)

    # _, acc = model.evaluate(X_test, Y_test, batch_size=args.batch_size)
    # print("Accuracy on the test set: %0.2f%%" % (100*acc))

    # Craft one specific attack type
    print(X_test.shape)
    print(Y_test.shape)
    X_normal,Y_normal,X_adv,Y_adv=craft_one_type(model, X_test, Y_test, args.attack)
    # 存入.csv文件

    print('Adversarial samples crafted and saved to %s ' % PATH_DATA)

    X_normal = np.array(X_normal)
    Y_normal = np.array(Y_normal)
    X_adv = np.array(X_adv)
    Y_adv = np.array(Y_adv)
    np.savetxt('.//data//X_normal.csv', X_normal, delimiter=',')
    np.savetxt('.//data//Y_normal.csv', Y_normal, delimiter=',')
    np.savetxt('.//data//X_adv.csv', X_adv, delimiter=',')
    np.savetxt('.//data//Y_adv.csv', Y_adv, delimiter=',')
    # _, adv_acc = model.evaluate(X_adv, Y_adv, batch_size=args.batch_size,
    #                         verbose=0)
    # print("Accuracy on the adv test set: %0.2f%%" % (100 * adv_acc))


    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either  'jsmf', or 'one-pixel' ",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=100)
    args = parser.parse_args()
    main(args)