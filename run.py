import NeuralNetwork
import numpy

numpy.set_printoptions(suppress=True)

with numpy.load('mnist.npz') as data:
    trainingInputs = data['training_images']
    trainingOutputs = data['training_labels']

trainingData = list(zip(trainingInputs, trainingOutputs))

layerSizes = (784,16,16,10)

net = NeuralNetwork.NeuralNetwork(layerSizes)

x = 1

before = net.predict(trainingInputs[x])
net.SGD(trainingData, 3, 10, 3.0)
after = net.predict(trainingInputs[x])
net.printAccuracy(trainingInputs, trainingOutputs)

print("Expected Result:")
print(trainingOutputs[x])
print("Before Training:")
print(before)
print("After Training:")
print(after)
