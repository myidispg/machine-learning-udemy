# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:25:57 2018

@author: Prashant Goyal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

# Cleaning the texts
import re
import nltk # library for nlp
nltk.download('stopwords')
# ^^  downloads a list of common words that are not relevant to reviews like this, that, prepositions etc
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# ^^ Stemming, keeping only root of the words. Loved changes to love, hated changes to hate etc.
corpus = []
for i in range(0, 1000):
    
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] )
    # ^^ Here we removed all the characters except a-z and A-Z and replaced them with a space.
    review = review.lower() # Converted all characters to lowercase
    review = review.split() # removed all spaces and returns a string
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = " ".join(review) # Convert list of processed words to a string seperated by space
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500) # This keeps only the 1500 most occuring words.
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting Logistic learning to the Training set
from sklearn.linear_model import LogisticRegression
classifier_logistics = LogisticRegression(random_state=0)
classifier_logistics.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_kernel_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_kernel_svm.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_gaussian = classifier.predict(X_test)
y_pred_logistics = classifier_logistics.predict(X_test)
y_pred_kernel_svm = classifier_kernel_svm.predict(X_test)
y_pred_random_forest = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian)
cm_logistics = confusion_matrix(y_test, y_pred_logistics)
cm_kernel_svm = confusion_matrix(y_test, y_pred_kernel_svm)
cm_random_forest = confusion_matrix(y_test, y_pred_random_forest)

# The best in this case is Naive Bayes