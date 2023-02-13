# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 06:00:56 2023

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

#Tratamientos de los NA's
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3 ] = imputer.transform(X[:, 1:3])
