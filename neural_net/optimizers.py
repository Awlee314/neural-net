

class SGD:
    """ 
    Stochastic Gradient Descent works as an optimizer by
    adjusting the weights in the opposite direction as the computed
    gradient. 

    """

    def __init__(self, lr):
        # lr is learning rate (typically between 0.01 to 0.1)
        # store learning rate
        self.lr = lr
    
    def step(self, parameters):
        # parameters are the (weight, gradient) tuples from the network.
        # For each weight we subtract lr * gradient and update
        for weight, gradient in parameters:
            # update each weight in place using numpys __isub__
            # this does not mutate the tuple (impossible) but changes the class which weight points to.
            weight -= self.lr * gradient

"""
Will add Adam optimizer once fully functional.
"""
