import numpy as np
import random
import NNlayers
import Hybrid

class perceptron:
    def __init__(self, activationFunction, hybridNeuronInPreviousLayer, numberOfPerceptronsInPreviousLayer = 0):
        self.value = 0
        self.activationFunction = activationFunction
        self.numberOfPerceptronsInPreviousLayer = numberOfPerceptronsInPreviousLayer
        self.numberOfHybridNeuronsInPreviousLayer = 0
        if numberOfPerceptronsInPreviousLayer:
            weights = [random.uniform(-0.5,0.5)]
            self.gradient_squared = [1]
            for _ in range(numberOfPerceptronsInPreviousLayer):
                x = random.uniform(-0.5,0.5)
                weights.append(x)
                self.gradient_squared.append(1)
            if hybridNeuronInPreviousLayer:
                x = random.uniform(-0.5,0.5)
                weights.append(x)
                self.gradient_squared.append(1)
                self.numberOfHybridNeuronsInPreviousLayer = 1
            self.weights = np.array(weights)
        self.differentialFunction = 1

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

    def setDeltaMid(self, nextLayer, index, hybridNeuron = None):
        self.setDifferentialFunction()
        summedError = 0
        for perceptron in nextLayer.perceptrons:
            summedError += perceptron.deltaValue * perceptron.weights[index+1]
        if hybridNeuron:
            summedError += hybridNeuron.deltaValue * hybridNeuron.getDifferential(index+1)
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
        for i in range(self.numberOfPerceptronsInPreviousLayer+1+self.numberOfHybridNeuronsInPreviousLayer):
            self.weights[i] += (learningRate * self.deltaValue * previousLayerValues[i]) 

    def ADAupdateWeights(self, learningRate, previousLayerValues):
        for i in range(self.numberOfPerceptronsInPreviousLayer+1+self.numberOfHybridNeuronsInPreviousLayer):
            dw = self.deltaValue * previousLayerValues[i]
            self.gradient_squared[i] += dw * dw
            self.weights[i] += (learningRate * self.deltaValue * previousLayerValues[i])/np.sqrt(self.gradient_squared[i])
    
    def RMSupdateWeights(self, learningRate, previousLayerValues, Beta):
        for i in range(self.numberOfPerceptronsInPreviousLayer+1+self.numberOfHybridNeuronsInPreviousLayer):
            dw = self.deltaValue * previousLayerValues[i]
            self.gradient_squared[i] = self.gradient_squared[i]*Beta + (1-Beta)*dw*dw
            self.weights[i] += (learningRate * self.deltaValue * previousLayerValues[i])/np.sqrt(self.gradient_squared[i])

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