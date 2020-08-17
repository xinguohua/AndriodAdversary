from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np



CHARACTERISTICS = ['kd', 'bua','bue','buc', 'lid']
PATH_DATA = "..//data//characteristic//"


#返回某一攻击某一特征的数据
def load_characteristics(attack, char):
    """
    :param attack:
    :param characteristics:
    :return:
    """
    X= None
    #打印分析的特征
    print(" Analyse -- %s" % char)
    #加载分析的特征文件
    file_name = os.path.join(PATH_DATA, "%s_%s.npy" % (char, attack))
    data = np.load(file_name)
    # 加载特征
    if X is None:
        X = data[:, :-1]
        num_samples = X.shape[0]
        partition = int(num_samples / 2)
        X_adv= X[:partition]
        X_norm= X[partition:]
    return X_adv, X_norm


def analysis(args):
    # 攻击参数在的范围
    assert args.attack in ['jsmf', 'deepfool', 'onefeature', 'fgsm'], \
        "Attack parameter must be either 'jsmf', 'deepfool', 'onefeature', 'fgsm'"



    #把特征拆开
    characteristics = args.characteristics.split(',')
    # 遍历所有的特征
    for char in characteristics:
        assert char in CHARACTERISTICS, \
            "Characteristic(s) to use 'kd', 'bua','bue','buc', 'lid'"
        # 返回对抗样本和正常样本的值
        X_adv, X_norm = load_characteristics(args.attack, char)
        # 对抗样本特征值-正常样本
        results=np.array(X_adv)-np.array(X_norm)
        lager=len(np.where(results > 0)[0])
        lagerrate=lager/len(X_adv)
        print("adv>norm",lagerrate)
        print("adv<norm",1-lagerrate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 选择对应的攻击（样本）
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'jsmf', 'deepfool', 'onefeature', 'fgsm' ",
        required=True, type=str
    )
    ## 选择要分析的特征
    parser.add_argument(
        '-r', '--characteristics',
        help="Characteristic(s) to use in ['kd', 'bua','buc','bue'] or combine ",
        required=True, type=str
    )


    args = parser.parse_args()
    analysis(args)
