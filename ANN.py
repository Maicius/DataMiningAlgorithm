#coding=utf-8
import math
import random
import re
import time
random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def createDataset():
    dataSet = []
    trainDataSet = []
    trainDataSetLabel = []
    testDataSet = []
    testDataSetLabel = []
    file = open("titanic.dat", "r")
    for line in file.readlines()[8:]:
        data =  re.split(',|', line)
        #remove '\r \n' in list
        data = map(lambda x: x.strip(), data)
        dataSet.append(data)
    #print dataSet
    file.close()
    for k in range(1400):
        trainData = []
        trainDataLabel = []
        dataSet[k][0] = (float(dataSet[k][0]) + 1.87)/(0.965 + 1.87)
        dataSet[k][1] = (float(dataSet[k][1]) + 0.228)/(4.38 + 0.228)
        dataSet[k][2] = (float(dataSet[k][2]) + 1.92)/(0.521 + 1.92)
        trainData.append(dataSet[k][0])
        trainData.append(dataSet[k][1])
        trainData.append(dataSet[k][2])
        trainDataSet.append(trainData)
        dataSet[k][3] = float(dataSet[k][3]) if float(dataSet[k][3]) > 0 else 0
        trainDataLabel.append(dataSet[k][3])
        trainDataSetLabel.append(trainDataLabel)
    for k in range(1401, 2200):
        testData = []
        testDataLabel = []
        dataSet[k][0] = (float(dataSet[k][0]) + 1.87) / (0.965 + 1.87)
        dataSet[k][1] = (float(dataSet[k][1]) + 0.228) / (4.38 + 0.228)
        dataSet[k][2] = (float(dataSet[k][2]) + 1.92) / (0.521 + 1.92)
        dataSet[k][3] = float(dataSet[k][3])
        testData.append(dataSet[k][0])
        testData.append(dataSet[k][1])
        testData.append(dataSet[k][2])
        testDataSet.append(testData)
        dataSet[k][3] = float(dataSet[k][3]) if float(dataSet[k][3]) > 0 else 0
        testDataLabel.append(dataSet[k][3])
        testDataSetLabel.append(testDataLabel)
    print "train："

    print "训练数据:"
    print trainDataSet

    print "训练数据标签:"
    print trainDataSetLabel

    print "测试数据:"
    print testDataSet

    print "测试数据标签:"
    print testDataSetLabel
    return trainDataSet, trainDataSetLabel, testDataSet, testDataSetLabel

class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        # print "output_cells:" + str(self.output_cells[:])
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error

        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error

        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        # print "error:" + str(error)
        return error

    def train(self, cases, labels, limit=1000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            print j
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self):
        trainDataSet, trainDataSetLabel, testDataSet, testDataSetLabel = createDataset()
        print "testData:" + str(testDataSet)
        self.setup(3, 4, 1)
        t0 = time.clock()
        print "训练中..."
        self.train(trainDataSet, trainDataSetLabel, 50000, 0.05, 0.1)
        print "训练完成, 耗时:"  + str(round(time.clock() - t0, 3)) + "秒"
        count = 0
        for i in range(len(testDataSet)):
            label = (self.predict(testDataSet[i]))
            print "预测结果:"+ str(label)
            print "标签:" + str(testDataSetLabel[i][0])
            if testDataSetLabel[i][0] - label[0] < 0.1 and testDataSetLabel[i][0] - label[0] > -0.1:
                count += 1
        print "正确率:" + str(round(count/len(testDataSet), 3))


if __name__ == '__main__':

    nn = BPNeuralNetwork()
    nn.test()
