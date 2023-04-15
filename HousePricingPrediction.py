!git clone "https://gitlab.com/ykristian/housing-dataset.git"
%cd housing-dataset
%ls

import pandas as pd
dataset = pd.read_csv("Housing.csv")

dataset.describe().T

dataset.info()

#Task 1: Remove a column that will has no effect as features for Sale Price prediction 
### Answer under this line
## SEBELUM
print(dataset)
## SESUDAH
dataset = dataset.drop(["Id"], axis = 1)
dataset = dataset.select_dtypes(exclude=['object'])
print(dataset)

#Task 2: Remove all columns that have missing value 
### Answer under this line
dataset = dataset.dropna(axis=1)

#Task 3: Separate the dataset into input (X) and target (y)
### Answer under this line
x = dataset.drop(["SalePrice"], axis = 1)
y = dataset["SalePrice"]


#Task 4: Perform Feature Scaling on input (X)
### Answer under this line
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)


#Task 5: Separate X and y into X_train, X_test, y_train and y_test (Using 75:25 ratio)
### Answer under this line
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size = 75, random_state=100)
print("Train X Shape = ", xtrain.shape, " Train Y Shape = ", ytrain.shape);
print("Test X Shape = ", xtest.shape, " Test Y Shape = ", ytest.shape);
print(xtrain)

#Task 5: Perform regression training on this dataset (You may use any regression technique)
### Answer under this line
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import math
## CLASSIFIER
M = MLPClassifier(hidden_layer_sizes=(50, 40, 20), verbose=1, tol=0.000001, max_iter=1000, n_iter_no_change=20)
M.fit(xtrain,ytrain)

#Task 6: Use your trained regression to predict X_test and capture the result in y_pred
### Answer under this line
ypred = M.predict(xtest)


#Task 7: Compare y_test and y_pred, show the mean squared error
### Answer under this line
import numpy as np
from sklearn.metrics import mean_squared_error
accuracy = math.floor(accuracy_score(test_y,ypred)
print("Accuracy Score : ", accuracy * 100), "%")
print(accuracy_score(ytest, ypred))