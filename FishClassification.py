!git clone "https://gitlab.com/ykristian/fish-data.git"
%cd fish-data/

import pandas as pd
dataset = pd.read_csv("Fish.csv")

dataset.info()
dataset.sample(10)

x =  dataset.drop(["Species"],axis=1)
y =  dataset["Species"]

y.hist()

y_new = pd.get_dummies(y)
print(y_new)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

x = scaler.transform(x)

print(x)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

clf = LogisticRegression(penalty = 'l1', solver = "saga", random_state = 1).fit(x,y)
y_pred = clf.predict(x)
print("Accuracy Score : ", accuracy_score(y,y_pred)*100, "%")

y_pred = model.predict(x)
print("Accuracy Score : ", accuracy_score(y,y_pred)*100, "%")

x_test = [[242.0,	23.2,	25.4,	30.0,	11.5,	4.5],[19,11,12,13,2,1] ]
c = scaler.transform(x_test)
ypred = model.predict(c)
print(ypred)

!pip install anvil-uplink

import anvil.server

anvil.server.connect("MMA3M52NFDNNLOOBVSPUHRS2-BT7TZ3RX7WOQZTLY")
@anvil.server.callable
def classify_ikan(a,b,c,d,e,f):
  return "Hello"

anvil.server.wait_forever()