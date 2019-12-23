import NeuralNetwork
import numpy
import random

numpy.set_printoptions(suppress=True)

with numpy.load('mnist.npz') as data:
    trainingInputs = data['training_images']
    trainingOutputs = data['training_labels']

trainingData = list(zip(trainingInputs, trainingOutputs))

layerSizes = (784,16,16,10)

net = NeuralNetwork.NeuralNetwork(layerSizes)

x = random.randint(0, 5000)

# net.SGD(trainingData, 30, 10, 3.0)
net.loadParameters()
# net.writeParameters()
after = net.predict(trainingInputs[x])

print("Actual Number:")
print(numpy.argmax(trainingOutputs[x]))
print("Network Prediction:")
print(numpy.argmax(after))
