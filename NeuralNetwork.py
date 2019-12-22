import numpy
import random

class NeuralNetwork:

    def __init__(self, layerSizes):
        weightShapes = [(a,b) for a,b in zip(layerSizes[1:], layerSizes[:-1])]
        self.numLayers = len(layerSizes)
        self.weights = [numpy.random.standard_normal(s) / s[1] ** .5 for s in weightShapes]
        self.biases = [numpy.zeros((s,1)) for s in layerSizes[1:]]

    def predict(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(numpy.matmul(w,a) + b)
        return a

    def printAccuracy(self, inputs, outputs):
        predictions = self.predict(inputs)
        correct = sum([numpy.argmax(a) == numpy.argmax(b) for a,b in zip(predictions, outputs)])
        print('{0}/{1} accuracy: {2}%'.format(correct, len(inputs), (correct / len(inputs)) * 100))

    def evaluate(self, testData):
        testResults = [(numpy.argmax(self.predict(x)), y) for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)

    def costDerivative(self, outputActivations, y):
        return (outputActivations - y)

    def backprop(self, x, y):
        nablaB = [numpy.zeros(b.shape) for b in self.biases]
        nablaW = [numpy.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.costDerivative(activations[-1], y) * sigmoidDerivative(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = sigmoidDerivative(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nablaB[-l] = delta
            nablaW[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (nablaB, nablaW)

    def updateMiniBatch(self, miniBatch, learningRate):
        nablaB = [numpy.zeros(b.shape) for b in self.biases]
        nablaW = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(x, y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]
        self.weights = [w - (learningRate / len(miniBatch)) * nw for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (learningRate / len(miniBatch)) * nb for b, nb in zip(self.biases, nablaB)]

    def SGD(self, trainingData, epochs, miniBatchSize, learningRate, testData=None):
        if testData:
            nTest = len(testData)
        for j in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k + miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, learningRate)
            if testData:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), nTest))
            else:
                print("Epoch {0} complete".format(j))

def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
