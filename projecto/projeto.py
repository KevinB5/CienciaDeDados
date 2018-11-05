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

#Leitura dos dados
green = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/Quality Assessment-Digital Colposcopy/green.csv')
hinselmann = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/Quality Assessment-Digital Colposcopy/hinselmann.csv')
schiller = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/Quality Assessment-Digital Colposcopy/schiller.csv')

aps_test = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/aps_failure/aps_failure_test_set.csv')
aps_training = pd.read_csv('C:/Users/kevin/Desktop/IST/Ciencia de Dados/projecto/aps_failure/aps_failure_training_set.csv')


