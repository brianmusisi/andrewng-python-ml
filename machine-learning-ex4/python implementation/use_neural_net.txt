import numpy as np  
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import neural_net

data = loadmat('digits.mat')
X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y)

theta1, theta2 = neural_net.fit_model(X_train, y_train)

y_pred = neural_net.predict(theta1, theta2, X_test, y_test)


