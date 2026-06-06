import numpy as np
from train import train
from optimizers import SGD
from losses import *
from network import *
from layers import *

if __name__ == "__main__":
    # X contains all combinations of XOR
    # 0 ^ 0, 0 ^ 1, 1 ^ 0, 1 ^ 1
    X = np.array([[0,0,1,1],[0,1,0,1]])
    # y contains the outcomes of the 4 possible operations
    y = np.array([0,1,1,0])
    model = Network([Linear(2,10), ReLU(), Linear(10,2), SoftMax()])
    sgd = SGD(lr=1)
    # 2 inputs to first linear layer then 4 inputs to next layer
    loss_func = CrossEntropyLoss()

    train(model, X, y, sgd, loss_func, epochs=500)

