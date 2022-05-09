import numpy as np
import random
import NNlayers
import Perceptron


class hybrid:
    def __init__(self, activationFunction, numberOfPerceptronsInLayer):
        self.value = 0
        self.activationFunction = activationFunction
        self.numberOfPerceptronsInLayer = numberOfPerceptronsInLayer
        self.layervalues = []
        self.value = 1
        self.deltaValue = 1
        self.differentialFunction = 1

    def getValue(self):
        return self.value

    def setHybridValue(self, valueArray):#Define Based on Type of Kernel
        self.valueArray = valueArray
        self.value = self.productSum()
        self.sigmoid()

    def setDelta(self, nextLayer):
        index = self.numberOfPerceptronsInLayer + 1
        self.setDifferentialFunction()
        summedError = 0
        for perceptron in nextLayer.perceptrons:
            summedError += perceptron.deltaValue * perceptron.weights[index]
        self.deltaValue = summedError * self.differentialFunction

    def getDifferential(self, sentValue):#Define Based on Type of Kernel
        return self.layer_sum - sentValue

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
 
    def productSum(self): 
        array_sum = 0
        for i in self.valueArray: 
            array_sum += i
        self.layer_sum = array_sum
        array_sum_square = array_sum * array_sum 
        # print(self.valueArray, self.layer_sum)
        individual_square_sum = 0
        for i in self.valueArray: 
            individual_square_sum += i * i             
        return (array_sum_square - individual_square_sum) / 2

