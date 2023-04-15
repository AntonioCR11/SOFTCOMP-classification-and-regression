# COLAB : https://colab.research.google.com/drive/1FhJFXipDXDDeN58bt5a6c7JU-5QMVLus

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd

# For ordinal encoding categorical variables, splitting data
from sklearn import *
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
dataset = pd.read_csv("../input/imdb-dataset-of-top-1000-movies-and-tv-shows/imdb_top_1000.csv")

dataset

# Insert different genre in an array
genreData = []
for i in range (len(dataset)):
    strGenre = dataset["Genre"][i].split(", ")
    for j in range(len(strGenre)):
        cond = 0
        for k in range(len(genreData)):
            if strGenre[j]==genreData[k]:
                cond = 1
        if cond == 0:
            genreData.append(strGenre[j])
genreData


# Check genre data from dataset
# Insert it in array 2D
from array import *
genreVal = []
# Loop for how many genre listed
for i in range(len(genreData)):
    # Loop for every genre in each row of dataset
    genreVal.append([])
    for j in range(len(dataset)):
        strGenre = dataset["Genre"][j].split(", ")
        cond = 0
        # Split genre in each row
        for k in range(len(strGenre)):
            if strGenre[k]==genreData[i]:
                cond = 1
        if cond == 1:
            genreVal[i].append(1)
        else :
            genreVal[i].append(0)


grossFloat = []
for str in dataset['Gross']:
    if(pd.isna(str)):
        grossFloat.append(float(0.0))
    else: 
        strToNum = str.replace(",","")
        grossFloat.append(float(strToNum))
df = pd.DataFrame({"Gross":grossFloat})
df['Gross'] = df['Gross'].astype('float64')
y = df['Gross']

dataset=dataset.fillna(0)
#df.replace(np.nan, 0)
#dataset=dataset['No_of_Votes'].replace(np.nan, 0)
#dataset=dataset['Gross'].replace(np.nan, 0)

# Drop Genre, Gross, Poster_Link
# Drop Genre Because there are new column and row for it
x = dataset.drop(['Poster_Link','Gross','Genre'], axis=1)
x.head()

import category_encoders as ce
encoder = ce.OneHotEncoder(use_cat_names=True)
x_encoded = encoder.fit_transform(x)
# Create new column and row for genre
for i in range(len(genreData)):    
    x_encoded[genreData[i]] = genreVal[i]
    
x_encoded



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.1,random_state=42)
x_train.head()
y_train.head()

import matplotlib.pyplot as plt
plt.hist(y)

import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x_encoded,y)
reg.score(x_encoded,y)

y_pred = reg.predict(x_encoded)

from sklearn.metrics import mean_squared_error
hsl=mean_squared_error(y_test, y_pred)

totalBenar=0

rata=0
arr=y_test.to_numpy()
for n in range(len(arr)):
  nltest=arr[n]
  nlpredict=y_pred[n]
    
  #print(nltest)
  #print(nlpredict)
  
  sls=abs(nltest-nlpredict)
  rata=rata+sls
  if (sls<0.3) :
    totalBenar=totalBenar+1
  #print(sls)  
  #print("      ")
  #print(arr[n])

print(str(totalBenar)+"/"+str(len(arr)))
rata=rata/len(arr)


sum = 0
for i in range(len(y)):
  print(y[i], " -- ", y_pred[i])
  dif = y[i]-y_pred[i]
  sum = sum+dif*dif 
avg = sum / len(y)
print(avg)

from sklearn.metrics import mean_squared_error
mean_squared_error(y, y_pred)