# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:05:02 2018

@author: Prashant Goyal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Getting the dataset
dataset = pd.read_csv('Market_Basket_optimisation.csv', header=None)

transactions = []

for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003 , min_confidence= 0.4, min_lift= 3, min_length=2) 

# Visualising the results
results = list(rules)

