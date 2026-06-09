import numpy as np

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
Adam is an optimization algorithm that combines the benefits of both Momentum and RMSProp. It maintains a moving average of the gradients and a moving average of the squared gradients (second moment). 
This could overcome the local minima problem and speed up convergence in training neural networks.
"""
class ADM:
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Lr is Learning rate
        self.lr = lr
        # beta 1 and beta 2 are decay rates for the moving averages of the gradients and squared gradients
        self.beta1 = beta1
        self.beta2 = beta2
        # epsilon is a small constant to prevent division by zero
        self.epsilon = epsilon
        # first, second moment, and timestep
        self.m = None
        self.v = None
        self.t = 0
    """
    Adam update rules
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    weight -= (lr / (sqrt(v_hat) + epsilon)) * m_hat

    """
    def step(self, parameters):
        if self.m is None:
            self.m = [np.zeros_like(w) for w, _ in parameters]
            self.v = [np.zeros_like(w) for w, _ in parameters]
            """Start all array with zeros of same shape as weights"""

        # update time step
        self.t += 1
        for i, (weight, gradient) in enumerate(parameters):
            # adam formula 
            # first = beta1 * m + (1 - beta1) * gradient
            # second = beta2 * v + (1 - beta2) * (gradient ** 2)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradient ** 2)
            # bias correction
            # first moment estimate m_hat = m / (1 - beta1^t)
            # second moment estimate v_hat = v / (1 - beta2^t)
            # for each weight, update using the Adam update rule

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            weight -= (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat