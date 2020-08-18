import random
import math
import numpy
class AnnealingFunctions(object):
    def __init__(self,obj):
        # 初始化背包问题对象
        self.obj = obj

    def getInitSolution(self,templist,doindex):
        '''Returns bit-array of length prob_size
        返回长度为prob_size的位数组
        返回问题规模的数组
        '''
        individual = templist.copy()
        # restrAPIcalls
        restrAPIcalls = list(range(22759, 22908))
        # suspAPIcalls networkaddress userpermissions
        others = list(range(24809, 25000))
        codeindex = restrAPIcalls + others
        found = False
        while found==False:
            index = random.randint(0,doindex[0].shape[0]-1)
            index=doindex[0][index]
            if templist[index] == 0 and index not in codeindex:
                individual[index] = 1
                found = True
        return individual,index


    def getNeighbouringSolution(self,individual,viewindex,doindex):
        # restrAPIcalls
        restrAPIcalls = list(range(22759, 22908))
        # suspAPIcalls networkaddress userpermissions
        others = list(range(24809, 25000))
        codeindex = restrAPIcalls + others
        NeighbouringSolution=individual.copy()
        found = False
        while found==False:
            index = random.randint(0,doindex[0].shape[0]-1)
            index=doindex[0][index]
            if individual[index] == 0 and index not in viewindex  and index not in codeindex:
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
