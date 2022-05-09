import numpy as np
import random
import NNlayers

class perceptron:
    def __init__(self, activationFunction, numberOfPerceptronsInPreviousLayer = 0):
        self.value = 0
        self.activationFunction = activationFunction
        self.numberOfPerceptronsInPreviousLayer = numberOfPerceptronsInPreviousLayer
        if numberOfPerceptronsInPreviousLayer:
            weights = [random.uniform(-0.5,0.5)]
            for _ in range(numberOfPerceptronsInPreviousLayer):
                x = random.uniform(-0.5,0.5)
                weights.append(x)
            self.weights = np.array(weights)

    def getValue(self):
        return self.value

    def setPerceptronValue(self, valueArray):
        self.value = 0
        if self.numberOfPerceptronsInPreviousLayer:
            self.value = np.dot(self.weights, valueArray)
            if self.activationFunction == "sigmoid":
                self.sigmoid()
            elif self.activationFunction == "relu":
                self.relu()
            elif self.activationFunction == "tanh":
                self.tanh()
        else:
            self.value = valueArray

    def setDeltaMid(self, nextLayer, index):
        self.setDifferentialFunction()
        summedError = 0
        for perceptron in nextLayer.perceptrons:
            summedError += perceptron.deltaValue * perceptron.weights[index+1]
        self.deltaValue = summedError * self.differentialFunction

    def setDeltaLast(self, expectedValue):
        self.setDifferentialFunction()
        self.deltaValue = (expectedValue - self.value)* self.differentialFunction

    def setDifferentialFunction(self):
        if self.activationFunction == "sigmoid":
            self.differentialFunction = (self.value * (1 - self.value))
        elif self.activationFunction == "relu":
            if self.value <= 0:
                self.differentialFunction = 1
            else:
                self.differentialFunction = 1
        elif self.activationFunction == "tanh":
            self.differentialFunction =  1 - (self.value * self.value)
      
    def updateWeights(self, learningRate, previousLayerValues):
        for i in range(self.numberOfPerceptronsInPreviousLayer+1):
            self.weights[i] += (learningRate * self.deltaValue * previousLayerValues[i]) 

    def sigmoid(self):
        self.value = 1 / (1 + np.exp(-1*self.value))
        
    def relu(self):
        if self.value < 0:
            self.value = 0
    
    def tanh(self):
        ez = np.exp(self.value)
        numerator = ez - (1/ez)
        denominator = ez + (1/ez)
        self.value = numerator/denominator 