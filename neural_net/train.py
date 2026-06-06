import numpy as np
import network as model
import optimizers as op
import layers as layers
from losses import *
import math

def train(model, X, y, optimizer, loss_function, 
          epochs = 16, batch_size = 32):
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
    # Use number of examples if smaller than batch size
    batch_size = min(batch_size, X.shape[1])
    losses = []
    # Find number of batches to make (need at least 1 batch)
    num_batches = math.ceil(X.shape[1] / batch_size)
    for epoch in range(1,epochs+1):
        # Makes an array of indices and randomly permutes
        indices = np.random.permutation(X.shape[1])
        X_shuffled = X[:,indices]
        y_shuffled = y[indices]
        epoch_loss = 0
        # this loop for each batch
        for i in range(num_batches):
            # Last index not included in slicing
            X_batch = X_shuffled[:,i*batch_size:(i+1)*batch_size]
            y_batch = y_shuffled[i*batch_size:(i+1)*batch_size]
            # Apply forward pass onto the split parts
            predictions = model.forward(X_batch)
            epoch_loss += loss_function.forward(predictions, y_batch) 
            '''comput the loss and adding all the loss from each batches '''
            # Backwards pass on split parts
            grad_out = loss_function.backward()
            model.backward(grad_out)
            # Update with optimizer
            optimizer.step(model.parameters())

        losses.append(epoch_loss / num_batches)
        if (epoch % 1 == 0):
            # Print average loss for all batches in this epoch
            if(isinstance(loss_function, CrossEntropyLoss)):
                correct_prob = np.exp(-epoch_loss / num_batches)
            print(f"Epoch {epoch}, Loss {epoch_loss / num_batches:.4f}, correct prob {correct_prob}")
        

    return losses
        



    
    

    