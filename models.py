import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def train_regression(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X_train, y_train)
    print (model)
    return(model)

def evaluation_of_regression(model, X_test, y_test):
    predictions = model.predict(X_test)
    np.set_printoptions(suppress=True)
    print('Predicted labels: ', np.round(predictions)[:10])
    print('Actual labels   : ' ,y_test[:10])
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Daily Bike Share Predictions')
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test,p(y_test), color='magenta')
    plt.show()
    
def train_binary_classification(model, X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    reg = 0.01
    model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)
    print (model)
    return(model)
    
def evaluation_of_binary_classification(model, X_test, y_test):
    predictions = model.predict(X_test)
    print('Predicted labels: ', predictions)
    print('Actual labels:    ' ,y_test)
    from sklearn.metrics import accuracy_score
    print('Accuracy: ', accuracy_score(y_test, predictions))