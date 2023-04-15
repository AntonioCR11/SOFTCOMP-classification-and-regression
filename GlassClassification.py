# PR Soft Computing (Glass Classification + Anvil Front-End)
# Anvil : https://anvil.works/build#clone:2SS75QFLSBBICLJD=7YCNDA7HKYPVUMASYIV4X6FD

# 1A Load data CSV dari UCI machine learning
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
dataset = pd.read_csv(url, delimiter=',', names=["Id", "RI", "Na", "Mg","Al","Si","K","Ca","Ba","Fe","Type"])
print(dataset)
dataset.sample(10)

# 1B Feature Scaling Input
x = dataset.drop(["Id","Type"],axis=1)
y = dataset["Type"]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

x = scaler.transform(x)
print(x)
print(x.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

clf = LogisticRegression(penalty = 'l1', solver = "saga", random_state = 1).fit(x,y)
y_pred = clf.predict(x)
print("Accuracy Score : ", accuracy_score(y,y_pred)*100, "%")


y_one_hot = pd.get_dummies(y)
print(y_one_hot)

# 1C Training MLP Classifier
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(50,120,30), verbose=1, tol=0.000001)
model.fit(x,y)

# 1D Testing dan Laporan Akurasi
y_pred = model.predict(x)
print("Accuracy Score : ", accuracy_score(y,y_pred)*100, "%")


# COBA PREDICT
x_test = [[1.51651 , 14.38 , 0.00 , 1.94 , 73.61 , 0.00 , 8.48,  1.57 , 0.0]]
c = scaler.transform(x_test)
ypred = model.predict(c)
print("Type Prediction : ",ypred)

!pip install anvil-uplink


import anvil.server

anvil.server.connect("3QFTSGBQPE65VZJTQ34V2BZV-2SS75QFLSBBICLJD")

@anvil.server.callable
def classify_glass(a,b,c,d,e,f,g,h,i):
  x_test = [[a , b , c , d , e , f , g,  h , i]]
  c = scaler.transform(x_test)
  ypred = model.predict(c)
  #print("Type Prediction : ",ypred)
  glass = ""
  if(ypred == 1):
    glass = "building_windows_float_processed"
  elif(ypred == 2):
    glass = "building_windows_non_float_processed"
  elif(ypred == 3):
    glass = "vehicle_windows_float_processed"
  elif(ypred == 4):
    glass = "vehicle_windows_non_float_processed"
  elif(ypred == 5):
    glass = "containers"
  elif(ypred == 6):
    glass = "tableware"
  elif(ypred == 7):
    glass = "headlamps"

  return glass

anvil.server.wait_forever()