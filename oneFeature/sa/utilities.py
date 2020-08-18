'''Some useful functions'''
import matplotlib.pyplot as plt
import csv
from os import listdir
from os.path import isfile, join
import os
import numpy as np
def readfile(filename):
    '''read file into array of shape (n,2)'''
    arr = []
    try:
        f = open(filename).readlines()
        for line in f:
            split_string = line.split()
            arr.append(float(split_string[0]))
    except Exception as e:
        print(e)
        print('here')
    return arr

def plotgraph(x,y,name,save_name):

    for i in range(len(x)):
        plt.plot(x[i],y[i])
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.title(name)
    #plt.show()
    plt.savefig('..//sa//Plots//FitnessVIterations_'+save_name+'.pdf')
    plt.gcf().clear()

#画每条曲线 再求平均
def plotavegraph(x,y,name,save_name):
    sum=np.zeros(len(x[0]))
    for i in range(len(x)):
        plt.plot(x[i],y[i])
        sum=sum+np.array(y[i])
    sum=sum/len(x)
    plt.plot(x[1], sum,linestyle=':',label = "ave",color='black')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.title(name)
    #plt.show()
    plt.savefig('..//sa//Plots//FitnessVIterations_'+save_name+'.pdf')
    plt.gcf().clear()

def plotsinglegraph(x,y,name,save_name):
    plt.plot(x,y)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.title(name)
    #plt.show()
    plt.savefig('..//sa//Plots//FitnessVIterations_'+save_name+'.pdf')
    plt.gcf().clear()


def write_file(data,path):
    with open(path,"a") as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        writer.writerows(data)

def read_csv(filename):
    with open(filename,'r') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
    return data



def getListofFiles(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    return files

#正则化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range