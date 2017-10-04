#coding=utf-8
from __future__ import division
import numpy as np
import re
import time


def computeEntropy(x, y):
    if x == 0 or y == 0:
        return 0
    else:
        return -np.log2(x) - np.log2(y)


def createDataset():
    dataSet = []

    file = open("titanic.dat", "r")
    for line in file.readlines()[8:]:
        data =  re.split(',|', line)
        #remove '\r \n' in list
        data = map(lambda x: x.strip(), data)
        dataSet.append(data)
    #print dataSet
    file.close()

    # 统计Class/ Age/ sex的属性有哪些值
    ClassVal = []
    AgeVal = []
    SexVal = []
    trueSum = 0
    falseSum = 0
    for k in range(2200):
        if dataSet[k][0] not in ClassVal:
            ClassVal.append(dataSet[k][0])
        if dataSet[k][1] not in AgeVal:
            AgeVal.append(dataSet[k][1])
        if dataSet[k][2] not in SexVal:
            SexVal.append(dataSet[k][2])
        # 统计数据集中1和-1的数量
        if float(dataSet[k][3]) == 1.0:
            trueSum += 1
        elif float(dataSet[k][3]) == -1.0:
            falseSum += 1
    ClassVal = sorted(ClassVal)
    AgeVal = sorted(AgeVal)
    SexVal = sorted(SexVal)
    # 计算信息熵
    EntropyS = computeEntropy(trueSum / 2200, falseSum / 2200)


    # 此处可知Age和Sex只有两种值，故接下来只计算Class的信息熵
    sumClassVal1 = 0
    sumClassVal2 = 0
    sumClassVal3 = 0
    sumClassVal0 = 0
    trueSum0 = 0
    falseSum0 = 0
    trueSum1 = 0
    falseSum1 = 0
    trueSum2 = 0
    falseSum2 = 0
    trueSum3 = 0
    falseSum3 = 0
    # 统计Class每个属性的数量

    for dataClass in dataSet:
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
    print "Class值的类型有:" + str(ClassVal)
    print "其中 -0.923有:" + str(sumClassVal0) + "个, -1.87有: " + str(sumClassVal1) + \
          "个, 0.024有: " + str(sumClassVal2) + "个, 0.965有: " + str(sumClassVal3)
    print "Age值的类型有:" + str(AgeVal)
    print "Sex值的类型有" + str(SexVal)

    # 计算每个属性的信息熵
    EntropyS0 = []
    EntropyS1 = []
    EntropyS2 = []
    EntropyS3 = []

    EntropyS0.append(ClassVal[0])
    EntropyS1.append(ClassVal[1])
    EntropyS2.append(ClassVal[2])
    EntropyS3.append(ClassVal[3])
    EntropyS0.append(computeEntropy(trueSum0 / sumClassVal0, falseSum0 / sumClassVal0))
    EntropyS1.append(computeEntropy(trueSum1 / sumClassVal1, falseSum1 / sumClassVal1))
    EntropyS2.append(computeEntropy(trueSum2 / sumClassVal2, falseSum2 / sumClassVal2))
    EntropyS3.append(computeEntropy(trueSum3 / sumClassVal3, falseSum3 / sumClassVal3))

    # 排序
    Entropy = []
    Entropy.append(EntropyS0)
    Entropy.append(EntropyS1)
    Entropy.append(EntropyS2)
    Entropy.append(EntropyS3)
    Entropy = sorted(Entropy, key=lambda x: x[1])
    # 计算信息增益最大的值
    gain = EntropyS - Entropy[0][1]
    print "最大信息增益为"+str(gain)+", 此时节点值为："+str(Entropy[0][0])

    # 根据阈值离散化数据并生成训练集和测试集
    trainDataSet = []
    testDataSet = []
    for i in range(2200):
        if i <= 1400:
            if dataSet[i][0] <= Entropy[0][0]:
                dataSet[i][0] = -1.0
            elif dataSet[i][0] > Entropy[0][0]:
                dataSet[i][0] = 1.0
            if dataSet[i][1] == AgeVal[0]:
                dataSet[i][1] = -1.0
            elif dataSet[i][1] == AgeVal[1]:
                dataSet[i][1] = 1.0
            if dataSet[i][2] == SexVal[0]:
                dataSet[i][2] = -1.0
            elif dataSet[i][2] == SexVal[1]:
                dataSet[i][2] = 1.0
            dataSet[i][3] = float(dataSet[i][3])
            trainDataSet.append(dataSet[i])
        else:
            if dataSet[i][0] <= Entropy[0][0]:
                dataSet[i][0] = -1.0
            elif dataSet[i][0] > Entropy[0][0]:
                dataSet[i][0] = 1.0
            if dataSet[i][1] == AgeVal[0]:
                dataSet[i][1] = -1.0
            elif dataSet[i][1] == AgeVal[1]:
                dataSet[i][1] = 1.0
            if dataSet[i][2] == SexVal[0]:
                dataSet[i][2] = -1.0
            elif dataSet[i][2] == SexVal[1]:
                dataSet[i][2] = 1.0
            dataSet[i][3] = float(dataSet[i][3])
            testDataSet.append(dataSet[i])
    print "训练数据:"
    print trainDataSet
    print "测试数据:"
    print testDataSet
    return trainDataSet, testDataSet


def navieBayes(trainDataSet, testDataset):

    # 一共有两个类别， 1 and -1, 先统计trainDataSet 中1和-1的数量
    countTrueVal = 0
    countFalseVal = 0
    trainDataSetlen = len(trainDataSet)
    countID = 0
    for trainData in trainDataSet:
        if trainData[3] == 1.0:
            countTrueVal += 1
        elif trainData[3] == -1.0:
            countFalseVal += 1
    countForcast = 0

    for testData in testDataset:
        countClassTrue = 0
        countClassFalse = 0
        countAgeTrue = 0
        countAgeFalse = 0
        countSexTrue = 0
        countSexFalse = 0
        countID += 1
        for trainData in trainDataSet:
            if trainData[3] == 1.0:
                if trainData[0] == testData[0]:
                    countClassTrue += 1
                if trainData[1] == testData[1]:
                    countAgeTrue += 1
                if trainData[2] == testData[2]:
                    countSexTrue += 1
            elif trainData[3] == -1.0:
                if trainData[0] == testData[0]:
                    countClassFalse += 1
                if trainData[1] == testData[1]:
                    countAgeFalse += 1
                if trainData[2] == testData[2]:
                    countSexFalse += 1
        PTrue = (countClassTrue / countTrueVal) * (countAgeTrue / countTrueVal) * (countSexTrue / countTrueVal) * \
                (countTrueVal / trainDataSetlen)

        PFalse = (countClassFalse / countFalseVal) * (countAgeFalse / countFalseVal) * \
                 (countSexFalse / countFalseVal) * (countFalseVal / trainDataSetlen)

        print str(countID) + ":\t类标为1的后验概率为:" + str(round(PTrue, 2)) + \
              "\t类标为-1的后验概率为:" + str(round(PFalse, 2))
        if PTrue >= PFalse:
            if testData[3] == 1.0:
                print str(testData) + "预测类标为:1, 预测正确"
                countForcast += 1
            else:
                print str(testData) + "预测类标为:1, 预测错误"
        else:
            if testData[3] == -1.0:
                print str(testData) + "预测类标为:-1, 预测正确"
                countForcast += 1
            else:
                print str(testData) + "预测类标为:-1, 预测错误"
        print '\n'
    print "训练完成，预测准确率为:" + str(countForcast / len(testDataset))

    return (countForcast / len(testDataset))


def result():
    t0 = time.clock()
    trainDataSet, testDataSet = createDataset()
    print "finish to create dataSet"
    print "begin..."
    successRate = navieBayes(trainDataSet, testDataSet)
    print "历时:" + str(round(time.clock() - t0, 3))+"秒"
    print "finish"
    print "预测成功率为："
    print successRate


if __name__ == '__main__':
    result()