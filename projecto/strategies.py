import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from classifiers import *
from metrics import *
from plots import *

def train_and_test_split(X,Y,train_sz):
	#Split the data in train and test
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = train_sz,random_state = 0)

	return X_train,Y_train,X_test,Y_test

def cross_val(X,Y,splits):
	accuracy_knn = []
	accuracy_nb = []
	accuracy_dt = []
	accuracy_rf = []
	#Split the data with k folds for train and the rest to test 
	skf = StratifiedKFold(n_splits = splits, random_state = 0, shuffle = True)
	#Split the data in the train and test
	for train_index, test_index in skf.split(X,Y):
	    X_train, X_test = X.loc[train_index], X.loc[test_index]
	    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]

	    pred_knn = KNN(X_train,Y_train,X_test,3)
	    acc_knn = accuracy(pred_knn,Y_test)
	    accuracy_knn.append(acc_knn)

	    pred_nb = NB(X_train,Y_train,X_test)
	    acc_nb = accuracy(pred_nb,Y_test)
	    accuracy_nb.append(acc_nb)

	    pred_dt = DT(X_train,Y_train,X_test,"gini")
	    acc_dt = accuracy(pred_dt,Y_test)
	    accuracy_dt.append(acc_dt)

	    pred_rf = RF(X_train,Y_train,X_test,10,"gini")
	    acc_rf= accuracy(pred_rf,Y_test)
	    accuracy_rf.append(acc_rf)

	df_accuracies = pd.DataFrame(index = np.arange(1,4),columns = ['KNN','NB','DT','RF'])
	df_accuracies.loc[:,'KNN'] = accuracy_knn
	df_accuracies.loc[:,'NB'] = accuracy_nb
	df_accuracies.loc[:,'DT'] = accuracy_dt
	df_accuracies.loc[:,'RF'] = accuracy_rf

	df_media = pd.DataFrame(index = np.arange(1,2),columns = ['KNN','NB','DT','RF'])
	df_media.loc[len(df_media)] = [np.mean(accuracy_knn),np.mean(accuracy_nb),np.mean(accuracy_dt),np.mean(accuracy_rf)]

	return df_accuracies,df_media