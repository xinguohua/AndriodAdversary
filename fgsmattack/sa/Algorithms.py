import random
from fgsmattack.sa import AnnealingFunctions
import time

class Algorithm:
    def __init__(self):
        self.name = 'Algorithm'
    def run(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def getName(self):
        return self.name

class SimulatedAnnealing(Algorithm):
    def __init__(self,max_iterations,temp_max,temp_change,KnapsackObj):
        self.ProblemSize = KnapsackObj.getProblemSize()
        self.MaxIterations = max_iterations
        self.MaxTemp = temp_max
        self.temp_change = temp_change
        #用背包问题初始化算法功能类
        self.AF = AnnealingFunctions.AnnealingFunctions(KnapsackObj)
        self.name = 'Simulated Annealing'

    def run(self,templist,perturbationsindex):
        start_time = int(round(time.time() * 1000))

        # 得到初始状态解决方案
        viewindex=[]
        current,index = self.AF.getInitSolution(templist,perturbationsindex)
        viewindex.append(index)


        # 把初始方案视为最好方案
        best = current.copy()

        #初始温度设置
        temp = self.MaxTemp

        x = [0]
        y = [self.AF.getValue(best)]

        operations = self.ProblemSize #initially generating a random solution
        for i in range(self.MaxIterations):
            #print("模拟退火内层"+str(i)+"次迭代")

            neigbourSolution,index = self.AF.getNeighbouringSolution(templist,viewindex,perturbationsindex)
            viewindex.append(index)
            #print("neigbourSolution"+str(neigbourSolution))
            #print("index位变化" + str(index))
            operations += index


            temp = self.AF.getTemperature(i, temp, self.temp_change)
            #print("temp 温度" + str(temp))

            siCost = self.AF.getValue(neigbourSolution)
            sCost = self.AF.getValue(current)
            operations += self.ProblemSize * 2  # two problems & fitness function depends on problem size

            #print("neigbourSolutionSicost" + str(siCost))
            #print("currentrSolutionScost" + str(sCost))
            randomnum = random.uniform(0, 1)
            if siCost <= sCost:
                # 跳到neigbourSolution
                #print("直接跳到neigbourSolution")
                #changes = changes + 1
                current = neigbourSolution
                if siCost <= self.AF.getValue(best):
                    #print("更新最优解")
                    best = neigbourSolution
                    #更新方案后最好方案的变换特征量
                    #bestchanges=changes
                operations += self.ProblemSize
            elif self.AF.getMonteCarlo(sCost, siCost, temp) > randomnum:
                # print("delta")
                # print(self.AF.getMonteCarlo(sCost, siCost, temp))
                # print("random")
                # print(randomnum)
                # print("概率跳到neigbourSolution")
                #changes = changes + 1
                current = neigbourSolution
            else:
                # print("delta")
                # print(self.AF.getMonteCarlo(sCost, siCost, temp))
                # print("random")
                # print(randomnum)
                # print("不跳")
                pass

            x.append(i+1)
            y.append(self.AF.getValue(current)) #for graph, don't add to operations
        end_time = int(round(time.time() * 1000))

        #内部迭代过程
        #最好方案 内部迭代坐标  最好方案改变特征数 操作 时间
        return best,x,y,operations,end_time-start_time




