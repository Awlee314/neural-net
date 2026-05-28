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
        """
        How loss changes with each weight
        output = W * x + b
        L is loss function
        grad_out is dL/d(output) flowing in from next layer backwards.
        Want dL/dW, dL/db, dL/dx (grad_W, grad_b, grad_input)
        Have dL/d(output) as grad_out
        d(output)/dW = x
        d(output)/db = 1
        d(output)/dx = W
        dL/dW = dL/d(output) * d(output)/dW -> grad_W
             so grad_out @ x = dL/d(output) * d(output)/dW
             -> grad_out @ x = dL/dW
             Need transpose to make the dimensions align
        dL/db = dL/d(output) * d(output)/db -> grad_b
             so grad_out @ 1 = grad_b
             sum over batch dimension to align
        dL/dx = dL/d(output) * d(output)/dx -> grad_input
             so grad_out @ W = grad_input
             Need transpose to make the dimensions align
        
        """


        self.grad_W = grad_out @ self.x.T  # dL/dW
        self.grad_b = np.sum(grad_out) # dL/db
        # Gradient to pass backwards
        return self.W.T @ grad_out # dL/dx

in_features = 2
out_features = 2
t = Linear(in_features, out_features)
print(t.W * np.sqrt(2/in_features))
print(t.b)
x = np.ones((2,1))
print(x)
print(t.forward(x))
