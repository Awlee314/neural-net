import numpy as np
import inspect


import layers as layers


class Network:

    def __init__(layers):
        # List of layers like:
        # [Linear(784,256), ReLU(), Linear(256,10), Softmax()]
        return 0
    
    def forward(x):
        # Loop through layers in order
        # Output of each layer becomes input of next
        # return final output
        
        # get all classes layers.py

        layers = [
            name for name, obj in inspect.getmembers(layers, inspect.isclass)
            if obj.__module__ == layers.__name__
        ]
        # start each layer with an instance of the class
        layer_instances = [cls() for cls in layers]
        
        for l in layer_instances:
            x = l.forward(x)
        output = x

        return output
    
    def backward(grad):
        # Loop through layers in reverse order
        # pass layer's gradients as the input to previous one
        return 0
    
    def paramters():
        # Through layers, find each linear layer and return list of (weight, gradient) tuples
        # optimizer uses these to know the updates
        return 0
    
    
