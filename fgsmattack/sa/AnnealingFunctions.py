import random
import math
import numpy
class AnnealingFunctions(object):
    def __init__(self,obj):
        # 初始化背包问题对象
        self.obj = obj

    def getInitSolution(self,templist,perturbationsindex):
        '''Returns bit-array of length prob_size
        返回长度为prob_size的位数组
        返回问题规模的数组
        '''
        individual = templist.copy()
        found = False
        while found==False:
            index = random.randint(0,len(perturbationsindex)-1)
            index=perturbationsindex[index]
            if templist[index] == 0:
                individual[index] = 1
                found = True
        return individual,index


    def getNeighbouringSolution(self,individual,viewindex,perturbationsindex):
        NeighbouringSolution=individual.copy()
        found = False
        while found==False:
            index = random.randint(0, len(perturbationsindex) - 1)
            index = perturbationsindex[index]
            if individual[index] == 0 and index not in viewindex:
                NeighbouringSolution[index] = 1
                found = True
        return NeighbouringSolution,index


    def getTemperature(self,i,temp,temp_change):
        if i==0:
            return temp
        else:
            return temp*temp_change

    def getMonteCarlo(self,current,si,temp):
        delta_cost = math.fabs(current - si)
        return numpy.exp(-delta_cost/temp)

    def getValue(self,solution):
        # 得到问题的fitness值
        return self.obj.fitness(solution)
