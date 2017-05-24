#coding=utf-8
from __future__ import division
import numpy as np
import re


def computeEntropy(x, y):
    if x == 0 or y == 0:
        return 0
    else:
        return -np.log2(x) - np.log2(y)

def createDataset():
    dataSet = []
    trainDataSet = []
    testDataSet = []
    trueSum = 0
    falseSum = 0
    ClassVal = []
    AgeVal = []
    SexVal = []
    sumClassVal1 = 0
    sumClassVal2 = 0
    sumClassVal3 = 0
    sumClassVal0 = 0

    Entropy = []
    EntropyS0 = []
    EntropyS1 = []
    EntropyS2 = []
    EntropyS3 = []
    trueSum0 = 0
    falseSum0 = 0

    trueSum1 = 0
    falseSum1 = 0

    trueSum2 = 0
    falseSum2 = 0

    trueSum3 = 0
    falseSum3 = 0

    file = open("titanic.dat", "r")
    for line in file.readlines()[8:]:
        data =  re.split(',|', line)
        #remove '\r \n' in list
        data = map(lambda x: x.strip(), data)
        dataSet.append(data)
    #print dataSet
    file.close()

    for k in range(1400):
        if dataSet[k][0] not in ClassVal:
            ClassVal.append(dataSet[k][0])
        if dataSet[k][1] not in AgeVal:
            AgeVal.append(dataSet[k][1])
        if dataSet[k][2] not in SexVal:
            SexVal.append(dataSet[k][2])

        if float(dataSet[k][3]) == 1.0:
            trueSum += 1
        elif float(dataSet[k][3]) == -1.0:
            falseSum += 1
        trainDataSet.append(dataSet[k])

    # 计算信息熵
    EntropyS = computeEntropy(trueSum / 1400, falseSum / 1400)

    print len(ClassVal)
    print len(AgeVal)
    print len(SexVal)
    EntropyS0.append(ClassVal[0])
    EntropyS1.append(ClassVal[1])
    EntropyS2.append(ClassVal[2])
    EntropyS3.append(ClassVal[3])
    # 此处可知Age和Sex只有两种值，故接下来只计算Class的信息熵
    for dataClass in trainDataSet:
        if dataClass[0] == ClassVal[0]:
            sumClassVal0 += 1
            if float(dataClass[3]) == 1.0:
                # print "dataClass[3] == 1\t" + str(dataClass[0] +"\t"+ str(dataClass[3])+"\t" + str(ClassVal[0]))
                trueSum0 += 1
            elif float(dataClass[3]) == -1.0:
                # print "dataClass[3] == -1\t" + str(dataClass[3])
                falseSum0 += 1

        elif dataClass[0] == ClassVal[1]:
            sumClassVal1 += 1
            if float(dataClass[3]) == 1.0:
                trueSum1 += 1
            elif float(dataClass[3]) == -1.0:
                falseSum1 += 1

        elif dataClass[0] == ClassVal[2]:
            sumClassVal2 += 1
            if float(dataClass[3]) == 1.0:
                trueSum2 += 1
            elif float(dataClass[3]) == -1.0:
                falseSum2 += 1

        elif dataClass[0] == ClassVal[3]:
            sumClassVal3 += 1
            if float(dataClass[3]) == 1.0:
                trueSum3 += 1
            elif float(dataClass[3]) == -1.0:
                falseSum3 += 1

    print "trueSum0 \t " + str(trueSum0)
    print "falseSum0 \t " + str(falseSum0)
    print "sumClassVal \t " + str(sumClassVal0)
    EntropyS0.append(computeEntropy(trueSum0 / sumClassVal0, falseSum0 / sumClassVal0))
    EntropyS1.append(computeEntropy(trueSum1 / sumClassVal1, falseSum1 / sumClassVal1))
    EntropyS2.append(computeEntropy(trueSum2 / sumClassVal2, falseSum2 / sumClassVal2))
    EntropyS3.append(computeEntropy(trueSum3 / sumClassVal3, falseSum3 / sumClassVal3))

    Entropy.append(EntropyS0)
    Entropy.append(EntropyS1)
    Entropy.append(EntropyS2)
    Entropy.append(EntropyS3)

    Entropy = sorted(Entropy, key = lambda x:x[1])

    print "Entropy"
    print Entropy

    print "class:"
    print ClassVal
    print "Age:"
    print AgeVal
    print "Sex:"
    print SexVal
    print EntropyS
    #for k in range(1401, 2200):

    #print "train："

    print "训练数据:"
    print trainDataSet
    print "测试数据:"
    print testDataSet
    return trainDataSet, testDataSet

def result():
    trainDataSet, testDataSet = createDataset()
    print "finish to create dataSet"

    print "begin..."
    successRate = 0
    print "finish"
    print "预测成功率为："
    print successRate


if __name__ == '__main__':
    result()