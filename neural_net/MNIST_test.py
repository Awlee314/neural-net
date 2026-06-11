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

    X = X/255
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train.T
    X_test = X_test.T
    print("Y test",y_test.shape)
    print("X test",X_test.shape)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model_sgd = Network([Linear(784,128), ReLU(), Linear(128,64), ReLU(), Linear(64,10), SoftMax()])
    model_adam = Network([Linear(784,128), ReLU(), Linear(128,64), ReLU(), Linear(64,10), SoftMax()])
    sgd = SGD(lr=0.001)
    ADAM = ADM(lr=0.001)
    # 2 inputs to first linear layer then 4 inputs to next layer
    loss_func = CrossEntropyLoss()

    lossforSGD = train(model_sgd, X_train, y_train, sgd, loss_func, epochs=23, batch_size=64)
    lossforADAM = train(model_adam, X_train, y_train, ADAM, loss_func, epochs=10, batch_size=64)
    correctCount = []
    TotalCount = []
    for i in range(len(y_test)):
        if i < 50:
            
            print(f"Expected digit is: {y_test[i]}")
            predicted = np.argmax(model_adam.forward(X_test[:,i:i+1]))
            print(f"Predicted digit is: {predicted}")
            TotalCount.append(1)
            if predicted == y_test[i]:
                correctCount.append(1)
    print(f"Accuracy of the model is: {sum(correctCount)/sum(TotalCount)*100}%")


    # plotting the loss for both optimizers
    plt.plot(lossforSGD, label='SGD')
    plt.plot(lossforADAM, label='ADAM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    
