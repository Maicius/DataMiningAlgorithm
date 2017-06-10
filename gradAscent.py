#coding=utf-8
import numpy as np
import math
import random

def sigmoid(x):
    return x

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
    labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error #matrix mult
    print "梯度下降:" + str(weights)
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix = np.mat(dataMatrix)
    m,n = np.shape(dataMatrix)

    alpha = 0.01
    weights = np.ones((n, 1))   #initialize to all ones
    for i in range(m):
        h = sigmoid(dataMatrix[i]*weights)
        error = classLabels[i] - h
        weights = weights + alpha * dataMatrix[i].transpose() * error
    print "随机梯度下降1:" + str(weights)
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones((n, 1))   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  #随机选取样本
            h = sigmoid((dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])#删除所选的样本
    print "随机梯度下降2:" + str(weights)
    return weights

if __name__ == '__main__':
    cases = [
        [1, 1],
        [1, 1],
    ]
    labels = [1, 1]
    gradAscent(cases, labels)
    stocGradAscent0(cases, labels)
    stocGradAscent1(cases, labels)