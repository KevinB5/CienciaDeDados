import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Leitura dos dados Digital Colposcopies
green = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/Quality Assessment-Digital Colposcopy/green.csv')
hinselmann = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/Quality Assessment-Digital Colposcopy/hinselmann.csv')
schiller = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/Quality Assessment-Digital Colposcopy/schiller.csv')

#Verificar se o formato é utilizável 
green.head()
hinselmann.head()
schiller.head()

#Remover os dados vazios
green = green.replace(0.0,np.nan)
green = green.dropna()
print green.isnull().sum()

hinselmann = hinselmann.replace(0.0,np.nan)
hinselmann = hinselmann.dropna()
print hinselmann.isnull().sum()

schiller = schiller.replace(0.0,np.nan)
schiller = schiller.dropna()
print schiller.isnull().sum()

#Leitura dos dados APS failure at Scania trucks
aps_test = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/aps_failure/aps_failure_test_set.csv')
aps_training = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/aps_failure/aps_failure_training_set.csv')



