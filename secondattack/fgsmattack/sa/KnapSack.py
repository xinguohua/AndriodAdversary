import math
import numpy as np
import math
class Knapsack(object):
    def __init__(self,originlist,values,model):
        self.originlist = originlist
        self.values = values
        self.model=model

    #得到问题的规模 线性解长度
    def getProblemSize(self):
        return len(self.values)



    def getValues(self):
        return self.values

    def getoriginList(self):
        return self.originlist




    def fitness(self,indiviual):
        '''Very simple fitness function'''
        #目标函数 计算当前列表与线性解列变的距离
        #显然是距离越小越好

        indiviual = np.array(indiviual)
        values = np.array(self.values)
        target = indiviual - values
        #加上模型
        x=np.mat(indiviual)
        predic=self.model.predict(x)
        fitness = np.linalg.norm(target.flatten())-math.pow(10,30)*predic[0][0]

        return fitness


