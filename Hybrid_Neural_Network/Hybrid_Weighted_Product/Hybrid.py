import numpy as np
import random
import NNlayers
import Perceptron
from copy import deepcopy

class hybrid:
    def __init__(self, activationFunction, numberOfPerceptronsInLayer):
        self.value = 0
        self.activationFunction = activationFunction
        self.numberOfPerceptronsInLayer = numberOfPerceptronsInLayer
        self.layervalues = []
        self.value = 1
        self.deltaValue = 1
        self.differentialFunction = 1
        weights = [random.uniform(-0.5,0.5)]
        self.gradient_squared = [1]
        for _ in range(numberOfPerceptronsInLayer):
            x = random.uniform(-0.5,0.5)
            weights.append(x)
            self.gradient_squared.append(1)
        self.weights = weights


    def getValue(self):
        return self.value

    def setHybridValue(self, valueArray):#Define Based on Type of Kernel
        self.layervalues = valueArray
        value = 1
        for i in range(1,len(valueArray)):
            value *= ((self.weights[i] * self.layervalues[i]) + 0.1)
        self.value = value

    def setDelta(self, nextLayer, learningRate):
        index = self.numberOfPerceptronsInLayer + 1
        self.setDifferentialFunction()
        summedError = 0
        for perceptron in nextLayer.perceptrons:
            summedError += perceptron.deltaValue * perceptron.weights[index]
        self.deltaValue = summedError * self.differentialFunction
        self.oldWeights = deepcopy(self.weights) 
        for i in range(1,self.numberOfPerceptronsInLayer+1):
            self.weights[i] += (learningRate * self.deltaValue * self.layervalues[i]) 

    def getDifferential(self, index):#Define Based on Type of Kernel
        if index:
            return ((self.value) * self.oldWeights[index]) / ((self.oldWeights[index] * self.layervalues[index]) + 0.1)
        return 0

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