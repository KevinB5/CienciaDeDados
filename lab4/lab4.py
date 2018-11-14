import numpy as np
import pandas as pd
from pre_processing import *
from strategies import *
from plots import *

splits = 10
#Leitura dos dados
cancer = pd.read_csv('breast_cancer.csv')
#Instancias 
X = cancer.iloc[:,:9]
#Classe correspondente aos dados
Y = cancer['Class']
# #Classes existentes
classes = np.unique(Y)

X_preprocessing = preprocessData(X)

classifier = "DC"

cross_val(X_preprocessing,Y,splits,classifier)

# s = Source(	)
	

# clf = DecisionTreeClassifier(random_state = 0, criterion = 'gini')

# print "Accuracy Mean Test =",np.mean(cross_val(preprocessing,Y,clf,splits))
# print "Standart Devision Test =",np.std(cross_val(preprocessing,Y,clf,splits))
# print "Best Accuracy Test =",np.max(cross_val(preprocessing,Y,clf,splits))
# dc_plot(range(2,12),cross_val(preprocessing,Y,clf,splits),'Accuracy for all instances','Number of instances','Accuracy')
# print "Accuracy Mean Train =",cross_val(preprocessing,Y,clf,splits)[3]
# print "Standart Devision Train =",cross_val(preprocessing,Y,clf,splits)[4]
# print "Best Accuracy Train =",cross_val(preprocessing,Y,clf,splits)[5]
