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

#Tratamientos de los NA's
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3 ] = imputer.transform(X[:, 1:3])

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

#Dividir el data set en conjunto de entrenamiento 
#y conjunto de testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)