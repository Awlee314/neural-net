import numpy as np
import network as model
import optimizers as op
import layers as layers
import losses as loss_function

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
    ...
    rng = np.random.default_rng()
    n = X.shape[0]
    Testsize = X.shape[1]
    temp = X.copy()
    for i in range(epochs):
        # shuffle colums
        rng.shuffle(temp, axis=1)
        result = np.array_split(temp, (Testsize // 32) + 1, axis=1)
        outp = []
        loss = np.zeros[len(y)]
        # this loop for each batch
        for part in result:
            cur_outp = model.forward(part)
            outp.append(cur_outp) 
            loss += loss_function.forward(cur_outp, y) 
            '''comput the loss and adding all the loss from each batches '''
            grad_out = loss_function.backward()
            model.backward(grad_out)
            op.step(model.parameters())

        if (i % 10 == 0):
            print(" this world has loss " + loss)
        


        



    
    

    