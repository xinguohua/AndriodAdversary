from secondattack.fgsmattack.sa import utilities as ut
from secondattack.fgsmattack.sa  import KnapSack as KS
import numpy as np
from secondattack.fgsmattack.sa import Algorithms
import os
#origin
class Main(object):
    #Main类的初始参数 线性解 原始解 外层迭代次数 模拟退火参数（内层迭代次数 初始温度 温度下降率） 最大改变特征
    def __init__(self,dataset,origin,model,num_iterations=10,
                SA_iterations=200,MaxTemp=200,TempChange=0.5):
        #加上模型
        self.model=model



        #线性解ndarray
        self.dataset = dataset
        #线性解ndarray到list
        self.values = self.dataset.tolist()
        #print("对抗样本对应的线性解")
        #print(self.values)

        #原始对抗样本
        self.origin = origin
        #原始对抗样本ndarray到list
        self.originlist = self.origin.tolist()
        # print("原始对抗样本")
        # print(self.originlist)



        #迭代次数
        #print("模拟退火算法外层迭代次数"+str(num_iterations))
        self.num_iterations = num_iterations

        # 模拟退火的参数
        self.SA_iterations = SA_iterations
        self.MaxTemp = MaxTemp
        self.TempChange = TempChange
        print("SA_iterations:"+str(SA_iterations)+"MaxTemp:"+str(MaxTemp)+"TempChange:"+str(TempChange))


        #Initialize problem
        # 初始化背包问题
        #参数 原始解 线性解
        self.myKnapSack = KS.Knapsack(originlist=self.originlist,values=self.values,model=self.model)

        #Initialize an algorithm
        self.algorithm = None

        # 模拟退火算法
        #初始算法 模拟退火参数（内层迭代次数 最大温度 温度变化） 背包问题
        self.algorithm = Algorithms.SimulatedAnnealing(max_iterations=self.SA_iterations, temp_max=self.MaxTemp,
                                                           temp_change=self.TempChange, KnapsackObj=self.myKnapSack)


    def Run(self,templist,iteration,perturbationsindex):
        allX,allY = [],[] #iterations,fitness
        v= []
        o = []
        t = []
        solutions = []
        for i in range(self.num_iterations):
            best,x,y,operations,runt = self.algorithm.run(templist,perturbationsindex)
            allX.append(x)
            allY.append(y)
            o.append(operations)
            t.append(runt)
            solutions.append(best)
            v.append(self.myKnapSack.fitness(best))


            # print("x"+str(x))
            # print("y" + str(y))
            # print("allX"+str(allX))
            # print("allY"+str(allY))
            # print("v" + str(v))




        #Plot
        # save_name = self.algorithm.getName()+"_"+str(self.num_iterations)+"_sa_x_fitnees_pic"+str(iteration)

        #迭代次数和fitness图
        # ut.plotgraph(allX,allY,self.algorithm.getName()+": Fitness over "+str(self.num_iterations)+" iteration(s)",save_name)


        '''saving results to csv'''
        data = list()
        best_index = np.argmin(v)
        print("最好方案"+str(v[best_index]))
        data.append(['Algorithm','Iteration','Best Value','Operations','Time (Milli)'])
        data.append([self.algorithm.getName(),self.num_iterations,v[best_index],o[best_index],t[best_index]])
        ut.write_file(data,"./"+self.algorithm.getName()+"_Results.csv")
        return v[best_index],solutions[best_index],allX[best_index],allY[best_index]


'''MAIN ENTRANCE HERE'''


# m = Main()
# m.Run()
# print('Done processing')