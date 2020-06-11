import numpy as np
import Perceptron
import random

class NNlayers:
    def __init__(self, numberOfPerceptron, learningRate, activationFunction = None, previousLayer=None):
        self.learningRate = learningRate
        self.numberOfPerceptron = numberOfPerceptron
        self.previousLayer = previousLayer
        self.numberOfPerceptronsInPreviousLayer = 0
        self.nextLayer = None
        if previousLayer:
            previousLayer.setNextLayer(self)
            self.numberOfPerceptronsInPreviousLayer = self.previousLayer.numberOfPerceptron
        self.perceptrons = []
        for _ in range(numberOfPerceptron):
            p = Perceptron.perceptron(activationFunction, self.numberOfPerceptronsInPreviousLayer)
            self.perceptrons.append(p)
    
    def setNextLayer(self,nextLayer):
        self.nextLayer = nextLayer
        self.numberOfPerceptronsInNextLayer = self.nextLayer.numberOfPerceptron

    def setPreviousLayerValues(self):
        self.previousLayerValues = self.previousLayer.getValues()

    def getValues(self):
        layerValues = [1]
        for node in self.perceptrons:
            layerValues.append(node.getValue())
        # print(layerValues)
        return np.array(layerValues)

    def updateLayerPerceptrons(self, NetworkInput = None):
        if self.previousLayer == None:
            i = 0
            for node in self.perceptrons:
                node.setPerceptronValue(NetworkInput[i])
                i += 1
            self.nextLayer.updateLayerPerceptrons()
        elif self.nextLayer:
            self.setPreviousLayerValues()
            for node in self.perceptrons:
                node.setPerceptronValue(self.previousLayerValues)
            self.nextLayer.updateLayerPerceptrons()
        else:
            self.setPreviousLayerValues()
            for node in self.perceptrons:
                node.setPerceptronValue(self.previousLayerValues)

    def backtrack(self, propagation = "GD", Beta = 0.9, expectedValues=None):
        if expectedValues and not self.nextLayer:
            index = 0
            for perceptron in self.perceptrons:
                perceptron.setDeltaLast(expectedValues[index])
                index += 1
        elif self.previousLayer and self.nextLayer:
            index = 0
            for perceptron in self.perceptrons:
                perceptron.setDeltaMid(self.nextLayer, index)
                index += 1
        if self.previousLayer:
            if propagation == "GD":
                for perceptron in self.perceptrons:
                    perceptron.updateWeights(self.learningRate, self.previousLayerValues)
            elif propagation == "ADA":
                for perceptron in self.perceptrons:
                    perceptron.ADAupdateWeights(self.learningRate, self.previousLayerValues)
            elif propagation == "RMS":
                for perceptron in self.perceptrons:
                    perceptron.RMSupdateWeights(self.learningRate, self.previousLayerValues, Beta)
            self.previousLayer.backtrack(propagation = propagation, Beta = Beta)

    def changelearningrate(self, newLearningRate):
        self.learningRate = newLearningRate
    
    def genomicweightchange(self, depth, node, weight, change):
        if depth == 0:
            selectedNode = node % self.numberOfPerceptron
            selectedweight = weight % self.numberOfPerceptronsInPreviousLayer
            oldWeight = self.perceptrons[selectedNode].weights[selectedweight]
            self.perceptrons[selectedNode].weights[selectedweight] += (self.learningRate * self.perceptrons[selectedNode].weights[selectedweight] * change)
            return oldWeight
        else:
            return self.previousLayer.genomicweightchange(depth - 1, node, weight, change)

    def revertweightchange(self, depth, node, weight, value):
        if depth == 0:
            selectedNode = node % self.numberOfPerceptron
            selectedweight = weight % self.numberOfPerceptronsInPreviousLayer
            self.perceptrons[selectedNode].weights[selectedweight] = value
        else:
            self.previousLayer.revertweightchange(depth - 1, node, weight, value)

    def resetlayer(self):
        if self.previousLayer:
            for perceptron in self.perceptrons:
                perceptron.gradient_squared = [1]*(perceptron.numberOfPerceptronsInPreviousLayer+1)
                for i in range(self.numberOfPerceptronsInPreviousLayer+1):
                    perceptron.weights[i] = random.uniform(-0.5,0.5)
            self.previousLayer.resetlayer()
            
    def get_random_weight(self, depth, node, weight):
        if depth == 0:
            selectedNode = node % self.numberOfPerceptron
            selectedweight = weight % self.numberOfPerceptronsInPreviousLayer
            oldWeight = self.perceptrons[selectedNode].weights[selectedweight]
            return oldWeight
        else:
            return self.previousLayer.get_random_weight(depth - 1, node, weight)