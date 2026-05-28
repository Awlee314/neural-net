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
        return cross_entropy_loss

    def backward(self):
        n = self.targets.size

        gradient = self.predictions.copy()
        # Subtract 1 from correct classes
        gradient = gradient[self.targets, np.arange(n)] - 1

        # average over all examples
        # gives gradient of loss with respect to output
        # grad_out as used in linear layer
        return gradient / n



class MSELoss:
    def __init__(self):
    
    def forward(self):
    
    def backward(self):
    
