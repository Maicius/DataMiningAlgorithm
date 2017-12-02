#coding=utf-8
from __future__ import division
from numpy import *
import time
import math
import re

def remove_blank(data):
    for item in data:
        item = item.strip()
    return data
def createDataset():
    dataSet = []
    trainDataSet = []
    testDataSet = []
    file = open("titanic.dat", "r")
    for line in file.readlines()[8:]:
        data =  re.split(',|', line)
        #remove '\r \n' in list
        data = map(lambda x: x.strip(), data)
        dataSet.append(data)
    #print dataSet
    file.close()
    for k in range(1400):
        dataSet[k][0] = (float(dataSet[k][0]) + 1.87)/(0.965 + 1.87)
        dataSet[k][1] = (float(dataSet[k][1]) + 0.228)/(4.38 + 0.228)
        dataSet[k][2] = (float(dataSet[k][2]) + 1.92)/(0.521 + 1.92)
        dataSet[k][3] = float(dataSet[k][3])
        trainDataSet.append(dataSet[k])
    for k in range(1401, 2200):
        dataSet[k][0] = (float(dataSet[k][0]) + 1.87) / (0.965 + 1.87)
        dataSet[k][1] = (float(dataSet[k][1]) + 0.228) / (4.38 + 0.228)
        dataSet[k][2] = (float(dataSet[k][2]) + 1.92) / (0.521 + 1.92)
        dataSet[k][3] = float(dataSet[k][3])
        testDataSet.append(dataSet[k])
    #print "train："

    print "训练数据:"
    print trainDataSet
    print "测试数据:"
    print testDataSet
    return trainDataSet, testDataSet


def caculateDiff(trainDataSet, testDataSet, k):
    success = 0
    count = 0
    print "训练数据:"
    print trainDataSet
    print "测试数据:"
    print testDataSet
    diffArray = []

    for testData in testDataSet:
        count  = count + 1
        print count
        for trainData in trainDataSet:
            #使用欧式方法计算距离
            diff = math.sqrt((testData[0] - trainData[0])**2 + (testData[1] - trainData[1])**2 + (testData[2] - trainData[2])**2)
            diffArray.append([diff, trainData[3]])
            #print diff
        #根据距离进行排序
        diffArray = sorted(diffArray, key = lambda x:x[0])
        countS = 0
        countF = 0
        for i in range(k):
            if diffArray[i][1] == 1.0:
                countS = countS + 1
            elif diffArray[i][1] == -1.0:
                countF =countF + 1
        if countS > countF:
            if testData[3] == 1.0:
                success = success + 1
        else:
            if testData[3] == -1.0:
                success = success + 1
    print "K值:" + str(k)
    print "测试数据共:"+ str(count) + "个"
    print "预测成功数据：" + str(success) + "个"
    return float(success) / float(count)


def result():
    t0 = time.clock()
    trainDataSet, testDataSet = createDataset()
    print "finish to create dataSet"
    print "Please input K:"
    k = raw_input()
    print "begin..."
    successRate = caculateDiff(trainDataSet, testDataSet, int(k))
    print "历时:" + str(round(time.clock() - t0, 3)) + "秒"
    print "finish"
    print "预测成功率为："
    print successRate

if __name__ == '__main__':
    result()