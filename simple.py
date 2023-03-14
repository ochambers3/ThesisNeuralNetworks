import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
data = pd.read_excel('data.xlsx', usecols="A,M")
np_data = np.array(data)

output = pd.read_excel('data.xlsx', usecols="N")
np_output = np.array(output)




#data = genfromtxt('iris.data', delimiter=',', dtype=None, encoding=None, usecols=(0,1,2,3))
#output = genfromtxt('iris.data', delimiter=',', dtype=None, encoding=None, usecols=(4))

#print(data)
#count = 0 #Adding a comment here
#for i in output:
#    if i == 'Iris-setosa':
#        output[count] = 1
#    elif i == 'Iris-versicolor':
#        output[count] = 2
#    else:
#        output[count] = 3
#    count += 1


data_train, data_test, labels_train, labels_test = train_test_split(np_data,np_output, test_size=0.2, random_state=1)


model = LinearRegression()
model.fit(data_train,labels_train)

y_pred = model.predict(data_test)

print("Coefficients: \n", model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(labels_test,y_pred))
print("Coefficient of determination %.2f:" % r2_score(labels_test,y_pred))
