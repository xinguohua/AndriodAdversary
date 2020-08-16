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

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np

EPOCH = 10
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

#训练数据
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
# 自定义训练数据集
MareWareDataset_from_csv =MareWareDataset('..//..//..//data//x_train01.csv','..//..//..//data//y_train01.csv',transformations)
# print(MareWareDataset_from_csv.__getitem__(index=1))
train_loader = torch.utils.data.DataLoader(
    dataset=MareWareDataset_from_csv,
    batch_size=BATCH_SIZE,
    shuffle=True)

MareWareTestset_from_csv =MareWareDataset('..//..//..//data//x_test01.csv','..//..//..//data//y_test01.csv',transformations)
test_loader = torch.utils.data.DataLoader(
    dataset=MareWareTestset_from_csv,
    batch_size=BATCH_SIZE,
    shuffle=True)



print('train data len: ', len(train_loader.dataset)) #train data len:
print('test data len: ', len(test_loader.dataset)) #test data len:

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



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

dnn=DNN()
dnn.apply(weight_init)
print(dnn)



optimizer = torch.optim.Adam(dnn.parameters(), lr=LR)
best_acc = 0

elapsed_time = 0
start_time = time.time()

for epoch in range(EPOCH):
    dnn.train()
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        #train_x,train_y 100  mu sigma 100 10
        train_x = train_x.clone().detach().float()
        mu, sigma = dnn(train_x)
        #10 100 10   10次取样 100个样本 10个结果
        prob_total = torch.zeros((NUM_SAMPLES, train_y.size(0), CLASS_NUM))
        for t in range(NUM_SAMPLES):
            # assume that each logit value is drawn from Gaussian distribution, therefore the whole logit vector is drawn from multi-dimensional Gaussian distribution
            epsilon = torch.randn(sigma.size()) #100*10
            logit = mu + torch.mul(sigma, epsilon) #100*10
            prob_total[t] = F.softmax(logit, dim=1)

        prob_ave = torch.mean(prob_total, 0) #100*10
        loss = F.nll_loss(torch.log(prob_ave), train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: ', epoch, '| batch: ', batch_idx, '| train loss: %.4f' % loss.data.numpy())


    dnn.eval()
    dnn.apply(apply_dropout)
    correct = 0
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
        # Aleatoric uncertainty is measured by some function of sigma_ave.
        # Epistemic uncertainty is measured by some function of prob_ave (e.g. entropy).

    accuracy = correct / float(len(test_loader.dataset))
    print('-> Epoch: ', epoch, '| test accuracy: %.4f' % accuracy)
    if accuracy > best_acc:
        best_acc = accuracy
        # 保存整个网络
        torch.save(dnn,".//model//dnn.pkl")





elapsed_time = time.time() - start_time
print('Best test accuracy is: ', best_acc)   # 0.9893
print('Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))


