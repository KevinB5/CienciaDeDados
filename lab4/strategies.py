from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from classifiers import *
from metrics import *
from plots import *

def train_and_test_split(X,Y,train_sz):
	#Split the data in train and test
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = train_sz,random_state = 0)

	return X_train,Y_train,X_test,Y_test

def cross_val(X,Y,splits,classifier):
	#Number of folds
	instances = 2
	#Split the data with k folds for train and the rest to test 
	skf = StratifiedKFold(n_splits = splits, random_state = 0, shuffle = True)
	#Split the data in the train and test
	for train_index, test_index in skf.split(X,Y):
	    X_train, X_test = X.loc[train_index], X.loc[test_index]
	    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]

	    if (classifier == 'DC'):
	    	prediction = dc(X_train,Y_train,X_test,'gini', split = instances)

		accuracies = accuracy(prediction,Y_test)
		errors = error(prediction,Y_test)
		classi_report(prediction,Y_test)
		instances = instances + 1

	dc_plot(range(2,instances),accuracies,"Evolution of Accuracies","Number of Instances","Accuracies score")
	dc_plot(range(2,instances),errors,"Evolution of Error","Number of Instances","Error")
	print prediction.shape