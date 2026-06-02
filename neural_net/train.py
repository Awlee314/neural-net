import numpy as np

def train(model, X, y, optimizer, loss_function, 
          epochs, batch_size):
    """
    For each epoch (pass through training dataset) we shuffle
    the data so that the neural net does not simply memorize the ordering.
    
    Process 32 examples at a time in 'mini-batches' 
    For each batch:
        - Run forward pass
        - Compute loss
        - Run backward pass
        - Call optimizer.step()
    We then track the average loss per epoch in a list,
    print the loss every few epochs to monitor the decrease in loss,
    return loss history for a plot.
    """
    ...