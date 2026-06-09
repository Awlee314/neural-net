import numpy as np
from train import train
from optimizers import ADM, SGD
from losses import *
from network import *
from layers import *
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == "__main__":

    X,y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    #print(y.shape)
    #print(X.shape)
    X = X/255
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train.T
    X_test = X_test.T
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = Network([Linear(784,128), ReLU(), Linear(128,64), ReLU(), Linear(64,10), SoftMax()])
    """No adam yet """
    sgd = SGD(lr=0.001)
    ADAM = ADM(lr=0.001)
    # 2 inputs to first linear layer then 4 inputs to next layer
    loss_func = CrossEntropyLoss()

    lossforSGD = train(model, X_train, y_train, sgd, loss_func, epochs=23, batch_size=64)
    lossforADAM = train(model, X_train, y_train, ADAM, loss_func, epochs=23, batch_size=64)
    # plotting the loss for both optimizers
    plt.plot(lossforSGD, label='SGD')
    plt.plot(lossforADAM, label='ADAM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    
