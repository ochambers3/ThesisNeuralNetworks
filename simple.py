import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn import preprocessing

import pandas as pd
data = pd.read_excel('Data, Small Numbers.xlsx', usecols="B:M", skiprows=2)
np_data = np.array(data)

output = pd.read_excel('Data, Small Numbers.xlsx', usecols="N", skiprows=2)
np_output = np.array(output)

#print(np_data)
#print("Seperation")
#print(np_output)


data_train, data_test, labels_train, labels_test = train_test_split(np_data,np_output, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(data_train,labels_train)


test_pred = model.predict([[1,0,1,0,0.387755102,1,0.729166667,0,1,0.638297872
,0,1]])
print("test_pred: ", test_pred)

y_pred = model.predict(data_test)
#print("y_pred: ", y_pred)
#print("labels_test: ", labels_test)

print("Coefficients: \n", model.coef_, " Intercept: ", model.intercept_)
normalized_arr = preprocessing.normalize(model.coef_)
print("Normal: ", normalized_arr)
print("Mean squared error: %.2f" % mean_squared_error(labels_test,y_pred))
print("Mean absolute percentage error: %.2f " % mean_absolute_percentage_error(labels_test,y_pred))
