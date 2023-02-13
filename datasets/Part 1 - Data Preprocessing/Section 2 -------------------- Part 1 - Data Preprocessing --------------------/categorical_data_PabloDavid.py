# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 06:00:03 2023

@author: Pablo Velasquez
"""

#Preprocesado

#Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importar el dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
#X2 = dataset.iloc[:, :-1]

y = dataset.iloc[:, 3].values


#Codificar datos categoricos
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
    
""" Ya no es necesario
labelEncoder_X = preprocessing.LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
"""

cT = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])]
                                      , remainder='passthrough')

X = np.array(cT.fit_transform(X), dtype=np.float)

labelEncoder_y = preprocessing.LabelEncoder()
y = labelEncoder_y.fit_transform(y)