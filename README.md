# Data_Science
1er Entrega - Obtención de un modelo y predicción

Se trabaja con un dataset que contiene información de personas, la idea es predecir si la persona tiene un salario anual mayor a 50K dólares. Para ello, se realiza un análisis exploratorio de los datos, se define si se está frente a un problema de regresión o clasificación, se escoge el mejor modelo entre distintas alternativas, se evalúan distintas métricas (f1_score, accuracy_score, cross_val_score), se determina el área bajo la curva ROC, se entrena el modelo seleccionado, y finalmente se obtiene la predicción buscada.

Dependencias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
