#coding=utf-8
from __future__ import division
from numpy import *
import operator
import re

def createDataset():
    i = 0
    data = []
    dataSet = []
    i=0
    j=4
    file = open("titanic.dat", "r")
    for line in file.readlines()[8:]:
        data =  re.split(',|', line)
        #remove '\r \n' in list
        data = map(lambda x: x.strip(), data)
        dataSet.append(data)
        i = i + 1
    print dataSet
    for k in range(i):

        dataSet[k][0] = float(dataSet[k][0])/(0.965 + 1.87)
        dataSet[k][1] = float(dataSet[k][1])/(4.38 + 0.228)
        dataSet[k][2] = float(dataSet[k][2])/(0.521 + 1.92)
    print "归一化："
    print dataSet

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))

    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # print normDataSet
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    # print normDataSet
    return normDataSet, ranges, minVals


def result():
    createDataset()
    datingDataMat, datingLabels = createDataset()  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    book = 199
    film = 4
    print 'watch book ' + '199' + ' film' + ' 4'
    print 'you are the person belong to'
    print classify((301, 4), normMat, datingLabels, 1)


if __name__ == '__main__':
    result()