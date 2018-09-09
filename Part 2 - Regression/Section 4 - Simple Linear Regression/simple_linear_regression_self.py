# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:14:10 2018

@author: Prashant Goyal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 

# Fitting simple linear regression on Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = regressor.predict(X_test)
y_pred1 = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience(Training Set)")
plt.xlabel('Years of experience')
plt.ylabel("Salary")
plt.show()

# Visualising the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Salary vs Experience(Test Set)")
plt.xlabel('Years of experience')
plt.ylabel("Salary")
plt.show()