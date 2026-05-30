import numpy as np
from layers import *

class Network:

    def __init__(self, layers):
        # List of layers like:
        # [Linear(784,256), ReLU(), Linear(256,10), Softmax()]
        # each one is a class
        self.layers = layers
    
    def forward(self,x):
        # Loop through layers in order
        # Output of each layer becomes input of next
        # return final output
               
        # get all classes layers.py
        for l in self.layers:
            x = l.forward(x)
        output = x

        return output
    
    def backward(self, grad):
        # Loop through layers in reverse order
        # pass layer's gradients as the input to previous one
        gradient_current = grad
        for layer in self.layers.reverse():
            # update next gradient to previous layers gradient
            gradient_current = layer.backward(gradient_current)
    
    def parameters(self):
        # Through layers, find each linear layer and return list of (weight, gradient) tuples
        # optimizer uses these to know the updates
        params_list = []
        for layer in self.layers:
            # Check if we have a linear layer
            if isinstance(layer, Linear):
                params_list.append((layer.W, layer.grad_W))
                params_list.append((layer.b, layer.grad_b))

        return params_list
    
    
