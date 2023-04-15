import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataset = pd.read_csv(url, delimiter=';')

print(dataset)

dataset.head(10)

y = dataset["quality"]
print(y)
print(y.shape)

x = dataset.drop("quality", axis=1)
print(x)
print(x.shape)
print(type(x))

import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x, y)
reg.score(x, y)

y_pred = reg.predict(x)

sum=0
for i in range(len(y)):
  print(y[i], " -- ", y_pred[i] )
  dif = y[i]-y_pred[i]
  sum=sum+dif*dif
avg = sum / len(y)
print(avg)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print("MSE : ",mean_squared_error(y, y_pred))
print("MAE : ",mean_absolute_error(y, y_pred))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaled_x = scaler.fit_transform(x)
#print(scaled_x)

# Regression After Scaling
regAfter = LinearRegression().fit(scaled_x, y)
print("New Reg Score : ",regAfter.score(scaled_x, y),"\n") 

scaled_dataset = scaler.fit_transform(dataset)
print("Scaled Dataset : ",scaled_dataset.shape)
# Potong data array2D [baris , kolom]
# [startBaris:endBaris , startKolom:endKolom]
x_scaled = scaled_dataset[:,0:-1]
y_scaled = scaled_dataset[:,-1:]
print("x_scaled : ",x_scaled.shape)
print("y_scaled : ",y_scaled.shape)

# kembalikan y shape ke array 1D
y_scaled = y_scaled.reshape(1599,)
print("Fixed y_scaled : ",y_scaled)