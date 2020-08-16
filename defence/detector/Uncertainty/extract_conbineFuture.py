# When modeling epistemic uncertainty and aleatoric uncertainty, we use MC dropout as well as loss attenuation to capture both model uncertainty and data uncertainty.
# We put distributions on both weights of the network and outputs of the network.


import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import  math

import pandas as pd




#############现在的方式
BATCH_SIZE = 150
LR = 0.001
CLASS_NUM = 2
NUM_SAMPLES = 50


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

#数据类
class MareWareDataset(Dataset):
    def __init__(self, csv_path,csvlabel_path,transforms=None):
        # stuff 一些初始化过程写在这里
        self.transforms = transforms
        # 读取 csv 文件
        self.datainfo = pd.read_csv(csv_path, header=None)
        self.data=np.asarray(self.datainfo.iloc[:,:])
        # label
        self.labelsinfo = pd.read_csv(csvlabel_path, header=None)
        self.labels=np.asarray(self.labelsinfo.iloc[:, 0])
        # 计算 length
        self.data_len = len(self.datainfo.index)
    def __getitem__(self, index):
        # print(self.labels.shape)
        # print(self.data.shape)
        single_malware_label = self.labels[index]
        malware_as_np = self.data[index][:]

        return (malware_as_np, single_malware_label)

    def __len__(self):
        # stuff 返回所有数据的数量
        return self.data_len

# 定义 transforms
transformations = transforms.Compose([transforms.ToTensor()])
# MareWareTestset_from_csv =MareWareDataset('.//datafinal//X_adv.csv','.//datafinal//Y_adv.csv',transformations)
MareWareTestset_from_csv =MareWareDataset('.//datafinal//X_normal.csv','.//datafinal//Y_normal.csv',transformations)

test_loader = torch.utils.data.DataLoader(
    dataset=MareWareTestset_from_csv,
    batch_size=BATCH_SIZE,
    shuffle=True)


print('test data len: ', len(test_loader.dataset))


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = nn.Sequential(    # input shape (1, 28, 28)
            nn.Linear(
                in_features=25000,
                out_features=200,
                bias=True
            ),                         # output shape (16, 28, 28)
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Linear(
                in_features=200,
                out_features=200,
                bias=True
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Linear(
                in_features=200,
                out_features=200,
                bias=True
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.linear = nn.Sequential(  # input shape (1, 28, 28)
            nn.Linear(
                in_features=200,
                out_features=CLASS_NUM*2,
                bias=True
            ),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x=F.softmax(x)
        logit = self.linear(x)
        mu, sigma = logit.split(CLASS_NUM, 1)
        return mu, sigma
dnn = torch.load('.//model//dnn.pkl')
elapsed_time = 0
start_time = time.time()


dnn.eval()
dnn.apply(apply_dropout)
correct = 0

aleatoric=[]
epistemic=[]
combine=[]
for batch_idx, (test_x, test_y) in enumerate(test_loader):
    test_x = test_x.clone().detach().float()
    prob_total = torch.zeros((NUM_SAMPLES, test_y.size(0), CLASS_NUM)) #10次取样*一批100个*10个结果
    sigma_total = torch.zeros((NUM_SAMPLES, test_y.size(0), CLASS_NUM)) #10*100*10
    for t in range(NUM_SAMPLES):
        test_mu, test_sigma = dnn(test_x)
        prob_total[t] = F.softmax(test_mu, dim=1)
        sigma_total[t] = test_sigma

    prob_ave = torch.mean(prob_total, 0) #100*10
    pred_y = torch.max(prob_ave, 1)[1].data.numpy() #100
    correct += float((pred_y == test_y.data.numpy()).astype(int).sum())

    sigma_ave = torch.mean(sigma_total, 0) #100*10

    # 张量转数组

    sigma_avearray = sigma_ave.detach().numpy()
    prob_avearray = prob_ave.detach().numpy()
    prob_totalarray=prob_total.detach().numpy()

    # Epistemic uncertainty is measured by some function of prob_ave (e.g. entropy).===mean
    # 求var(mean)
    meanvar100=[]
    for p in prob_avearray:
        #定义熵
        H=0
        for i in range(2):
            H+=-1*p[i]*math.log(p[i])
        # print(H)
        #加到每个batch的认知不确定性集合里 方便一会儿combine
        meanvar100.append(H)
        #加到整个认知不确定集合里
        epistemic.append(H)
    # test_uncerts1 = prob_totalarray.var(axis=0)  # var(mean) 100*10
    # meanvar100 = test_uncerts1.mean(axis=1)  # var在平均 100
    # for meanvar in meanvar100:
    #     epistemic.append(meanvar)

    for id,z in enumerate(prob_avearray):
        # Aleatoric uncertainty is measured by some function of sigma_ave
        x=sigma_avearray[id] #mean(var)
        #varmean = np.abs(x).mean() #一个向量各个值代表方差平均  不同值平均
        varmean = np.abs(x)[1]  # 一个向量各个值代表方差平均  取恶意分类的方差
        # print(varmean)
        aleatoric.append(varmean)

        uncert=varmean+meanvar100[id]
        combine.append(uncert)




#存入.csv文件
#存入adv
# aleatoric=pd.DataFrame(data=aleatoric,columns=["aleatoric",])#
# aleatoric.to_csv(".//featuresfinal//x_adv_aleatoric.csv",encoding='gbk')
#
# epistemic=pd.DataFrame(data=epistemic,columns=["epistemic",])
# epistemic.to_csv(".//featuresfinal//x_adv_epistemic.csv",encoding='gbk')
# combine=pd.DataFrame(data=combine,columns=["combine",])
# combine.to_csv(".//featuresfinal//x_adv_combine.csv",encoding='gbk')


#=======存入normal================
aleatoric=pd.DataFrame(data=aleatoric,columns=["aleatoric",])#
aleatoric.to_csv(".//featuresfinal//x_normal_aleatoric.csv",encoding='gbk')

epistemic=pd.DataFrame(data=epistemic,columns=["epistemic",])
epistemic.to_csv(".//featuresfinal//x_normal_epistemic.csv",encoding='gbk')
combine=pd.DataFrame(data=combine,columns=["combine",])
combine.to_csv(".//featuresfinal//x_normal_combine.csv",encoding='gbk')




elapsed_time = time.time() - start_time

print('Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))








