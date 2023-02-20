"""
the canonical example of a function that cant be
learned with a simple linear model is XDR
"""
import numpy as np
form noahnet.train import train
from noahnet.nn import NeuralNet
from noahnet.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output size=2),
    Tanh(),
    Linear(input_size=2, output_size=2 )
])
train(net, inputs, targets)

for x, y in  zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
