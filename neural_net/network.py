import numpy as np


class Network:

    def __init__(layers):
        # List of layers like:
        # [Linear(784,256), ReLU(), Linear(256,10), Softmax()]
        return 0
    
    def forward(x):
        # Loop through layers in order
        # Output of each layer becomes input of next
        # return final output
        return 0
    
    def backward(grad):
        # Loop through layers in reverse order
        # pass layer's gradients as the input to previous one
        return 0
    
    def paramters():
        # Through layers, find each linear layer and return list of (weight, gradient) tuples
        # optimizer uses these to know the updates
        return 0
    
    
