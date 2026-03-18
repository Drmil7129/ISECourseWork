import numpy as np
import math
np.random.seed(0)

class Neuron:
    def __init__(self, weights,inputs,bias):
        self.weights = weights
        self.bias = bias
        self.inputs = inputs

    def output(self):
        return np.dot(self.inputs,np.array(self.weights).T) + self.bias

    def setWeights(self,weights):
        self.weights = weights

    def setBias(self,bias):
        self.bias = bias

    def setInputs(self,inputs):
        self.inputs = inputs

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def __init__(self):
        self.exp_values = []
        self.norm_values = []
        self.output = []
    def forward(self,outputs):
        self.exp_values = np.exp(outputs-np.max(outputs,axis=1,keepdims=True))
        self.output = self.exp_values / np.sum(self.exp_values,axis=1,keepdims=True)

X = [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12]]

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,5)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)
