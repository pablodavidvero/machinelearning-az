# -*- coding: utf-8 -*-
"""
Spyder Editor

@autor: Pablo David Velasquez
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

#Dividir el data set en conjunto de entrenamiento 
#y conjunto de testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""