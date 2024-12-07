# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:42:04 2024

@author: haide
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
from IPython.display import display, HTML

df = pd.read_csv('tsa_claims.csv')

df.info()
df.describe()