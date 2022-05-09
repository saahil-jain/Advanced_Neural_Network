import Perceptron
import NNlayers
import random

class Network:
    def __init__(self, numberOfPerceptronsInInputLayer, learningRate, propagation = "GD", Beta = 0.9, HybridNeuron = False):
        self.layers = []
        self.numberOfLayers = 1
        self.learningRate = learningRate
        self.propagation = propagation
        self.Beta = Beta
        layer = NNlayers.NNlayers(numberOfPerceptronsInInputLayer ,self.learningRate, HybridNeuron)
        self.layers.append(layer)
        self.oldvalues = {}
    
    def addlayer(self, numberOfPerceptrons, activationFunction = None, HybridNeuron = False):
        layer = NNlayers.NNlayers(numberOfPerceptrons, self.learningRate, HybridNeuron, activationFunction, self.layers[self.numberOfLayers-1])
        self.layers.append(layer)
        self.numberOfLayers += 1

    def train(self, inputValues, outputValues):
        self.layers[0].updateLayerPerceptrons(inputValues)
        # out = self.layers[self.numberOfLayers - 1].getValues()
        # print("expected Values : {:6.4f}".format(outputValues[0]), "\toutput : {}".format(out[1:]))
        self.layers[self.numberOfLayers - 1].backtrack(self.propagation, self.Beta, outputValues)

    def predict(self, inputValues):
        self.layers[0].updateLayerPerceptrons(inputValues)
        return self.layers[self.numberOfLayers - 1].getValues()[1:]

    def printweights(self):
        for layer in self.layers:
            for node in layer.perceptrons:
                if node.numberOfPerceptronsInPreviousLayer:
                    print("bias : ", "{:10.6f}".format(node.weights[0]), "\tweights : ", "{}".format(node.weights[1:]))
            print("\n")
    
    def changelearningrate(self, newLearningRate):
        self.learningRate = newLearningRate
        self.layers[self.numberOfLayers - 1].changelearningrate(newLearningRate)

    def revertweightchange(self):
        for key in self.oldvalues.keys():
            oldvalue = self.oldvalues[key]
            depth = key[0]
            node = key[1]
            weight = key[2]
            self.layers[self.numberOfLayers - 1].revertweightchange(depth, node, weight, oldvalue)
        
    def genomicweightchange(self, n):
        for _ in range(n):
            depth = random.randint(0,self.numberOfLayers-2)
            node = random.randint(0,50)
            weight = random.randint(0,50)
            change = random.randint(0,1)
            if change == 0:
                change = -1
            oldvalue = self.layers[self.numberOfLayers - 1].genomicweightchange(depth, node, weight, change)
            self.oldvalues[(depth, node, weight)] = oldvalue
            
    def confirmweightchange(self):
        self.oldvalues = {}

    def globalreset(self):
        self.layers[self.numberOfLayers - 1].resetlayer()

    def simulatedweightchange(self, depth, node, weight, change):
        if change == 0:
            change = -1
        oldvalue = self.layers[self.numberOfLayers - 1].genomicweightchange(depth, node, weight, change)
        self.oldvalues[(depth, node, weight)] = oldvalue
                        
    def get_random_weight(self):
        depth = random.randint(0,self.numberOfLayers-2)
        node = random.randint(0,50)
        weight = random.randint(0,50)
        oldvalue = self.layers[self.numberOfLayers - 1].get_random_weight(depth, node, weight)
        return (depth, node, weight, oldvalue)

    def set_random_weight(self, depth, node, weight, value):
        self.layers[self.numberOfLayers - 1].revertweightchange(depth, node, weight, value)