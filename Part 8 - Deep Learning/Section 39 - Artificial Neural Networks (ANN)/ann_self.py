# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:31:59 2018

@author: Prashant Goyal
"""

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1- Data Prepocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# Gender is not onehotencoded because that will generate 2 columns and one column will be removed to prevent dummy variable trap.
# So, we will be left with one column only.

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=11))
# input_dim was required because this is the first layer. Consecutive layers are adjusted automatically.
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
# if the output has more than 2 nodes, use soft_max instead of sigmoid
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# if the output has more than 2 outcomes, loss will be categorical_crossentropy
# adam is a stochastic gradient descent algorithm 

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)