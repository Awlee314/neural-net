import numpy as np

class CrossEntropyLoss:
    
    def __init__(self):
        self.targets = None
        self.predictions = None

    # Heavily penalizes high losses
    def forward(self, predictions, targets):
        # predictions are softmax outputs
        # targets are the batch size -> labels like [3, 1, 7, ...] which are correct labels for each example
        # e.g. example 0 is label 3, example 1 is label 1 etc.
        n = targets.size # number of instances
        # Pick out the row of which label should have and the column for each example
        correct_predictions = predictions[targets, np.arange(n)] 
        # Compute multiclass cross entropy loss
        # add 1e-9 to avoid log(0)
        cross_entropy_loss = -(np.log(correct_predictions+1e-9)).mean()

        self.predictions = predictions
        self.targets = targets
        # average loss per example
        return cross_entropy_loss

    def backward(self):
        n = self.targets.size

        gradient = self.predictions.copy()
        # Subtract 1 from correct classes
        # Incorrect classes keep their value
        # When we subtract 1 from the correct classes we are subtracting the one-hot target
        # Gives dL/dz as dL/dp * dp/dz
        gradient[self.targets, np.arange(n)] -= 1

        # average gradient per example with / n
        # gives gradient of loss with respect to output
        # grad_out as used in linear layer
        return gradient / n



class MSELoss:
    
    
    def __init__(self):
        self.predictions = None
        self.targets = None
    # get predictions from previous layer, target that we wish to get

    def forward(self, predictions, targets):
        # find difference between predction and target value
        self.predictions = predictions
        self.targets = targets
        n = len(targets)
        total = 0
        for i in range(n):
            error = predictions[i] - targets[i]    
            # squre it to elimate negataives

            squared_error = error * error          
            total = total + squared_error
        
        # Calculate the mean
        loss = total / n
        
        return loss
            
    
    def backward(self):
        n = len(self.targets)
        gradients = []
        # The partial derivative dL/dp[i], derived on paper above:
        #     dL/dp[i] = 2 * (p[i] - t[i]) / n
        for i in range(n):
            error = self.predictions[i] - self.targets[i]
            gradient = (2 * error) / n
            gradients.append(gradient)
        return gradients
    
