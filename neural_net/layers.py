import numpy as np

class Linear:


    def __init__(self, in_features, out_features):
        """
        Size of features inputted into the layer is in_features and expected feature output size
        is out_features.
        Thus a matrix W must be created that is of size in_features x out_features
        """
        # He initialization of weights
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2/in_features)
        # Initialize a bias for each in_feature to all zero
        # Bias same dimension as output features since it is applied to W*x
        self.b = np.zeros((out_features,1), dtype=float)
        self.grad_W = None
        self.grad_b = None


    def forward(self,x):
        # Cache input for backward pass weight gradient
        self.x = x
        res = self.W @ self.x + self.b
        return res
    def backward(self,grad_out):
        # How loss changes with each weight
        # grad_out is dL/d(output) flowing in from next layer backwards.
        self.grad_W = grad_out @ self.x.T # dL/dW
        self.grad_b = np.sum(grad_out) # dL/db
        # Gradient to pass backwards
        return self.W.T @ grad_out

in_features = 2
out_features = 2
t = Linear(in_features, out_features)
print(t.W * np.sqrt(2/in_features))
print(t.b)
x = np.ones((2,1))
print(x)
print(t.forward(x))


class ReLU:

    def __init__(self):
        self.x = None

    def forward (self, x):
        self.x = x
        return np.maximum(0, x)
    # When the output is greater than 0, return 1. Otherwise, return 0.
    def backward (self,grad_output):
        return grad_output * (self.x > 0)
    

    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    # derivative of the orginal function 
    def backward(self, grad_output):
        
        return grad_output * self.out * (1 - self.out)