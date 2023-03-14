import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
data = pd.read_excel('data.xlsx', usecols="B:M", skiprows=2)
np_data = np.array(data)

output = pd.read_excel('data.xlsx', usecols="N", skiprows=2)
np_output = np.array(output)

#print(np_data)
#print("Seperation")
#print(np_output)


data_train, data_test, labels_train, labels_test = train_test_split(np_data,np_output, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(data_train,labels_train)

r_sq = model.score(data_train,labels_train)
print("Training coef of d: ", r_sq)


#test_pred = model.predict([[1,0,1,0,42,22,28,1,0,26,0,1,]])
#print("test_pred: ", test_pred)

y_pred = model.predict(data_test)
#print("y_pred: ", y_pred)
#print("labels_test: ", labels_test)

print("Coefficients: \n", model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(labels_test,y_pred))
print("Coefficient of determination %.2f:" % r2_score(labels_test,y_pred))
