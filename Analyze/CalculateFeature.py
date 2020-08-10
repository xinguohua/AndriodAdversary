import pandas as pd
import numpy as np

#统计特征个数（全部和恶意软件）  求平均数 分位数

data = np.loadtxt(open("..//data//x_train01.csv","rb"),delimiter=",",skiprows=0)
labels = np.loadtxt(open("..//data//y_train01.csv","rb"),delimiter=",",skiprows=0)
data=np.array(data)
labels=np.array(labels)

####提取所有软件特征个数
FeatureNumberlist=[]
####提取恶意软件特征个数
MalwareFeatureNumberlist=[]
for i,sample in enumerate(data):
    FeatureNumberlist.append(np.sum(sample==1))
    if labels[i]==1:
        MalwareFeatureNumberlist.append(np.sum(sample==1))
FeatureNumberlist=np.array(FeatureNumberlist)
allmean=np.mean(FeatureNumberlist)
lower_q=np.quantile(FeatureNumberlist,0.25,interpolation='lower')#下四分位数
higher_q=np.quantile(FeatureNumberlist,0.75,interpolation='higher')#上四分位数
media=np.median(FeatureNumberlist) #中位数
# [33 29 44 ... 26 20 42] 29 87.94051483354721 76 42.0
print(FeatureNumberlist,lower_q,allmean,higher_q,media)

MalwareFeatureNumberlist=np.array(MalwareFeatureNumberlist)
allmean1=np.mean(MalwareFeatureNumberlist)
lower_q1=np.quantile(MalwareFeatureNumberlist,0.25,interpolation='lower')#下四分位数
higher_q1=np.quantile(MalwareFeatureNumberlist,0.75,interpolation='higher')#上四分位数
media=np.median(FeatureNumberlist) #中位数
print(MalwareFeatureNumberlist,lower_q1,allmean1,higher_q1,media)
# [33 29 44 ... 26 20 42] 26 41.605335082540726 49 42.0
