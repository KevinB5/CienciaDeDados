import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Classifier KNN
def KNN(X_train,Y_train,X_test,n):
	classifier = KNeighborsClassifier(n_neighbors = n)
	train = classifier.fit(X_train,Y_train)
	return train.predict(X_test)


#Classifier Naive Bayes
def NB(X_train,Y_train,X_test):
	classifier = GaussianNB()
	train = classifier.fit(X_train,Y_train)
	return train.predict(X_test)


#Classifier Decision Tree
def DT(X_train,Y_train,X_test,criterion,min_s = 2,max_d = None,leaf_nodes = None):
	classifier = DecisionTreeClassifier(criterion = criterion, min_samples_split = min_s, max_depth = max_d, random_state = 0, max_leaf_nodes = leaf_nodes)
	train = classifier.fit(X_train,Y_train)
	return train.predict(X_test)


#Classifier Random Forest
def RF(X_train,Y_train,X_test, n_estim, criterion,min_samples, max_depth):
	classifier = RandomForestClassifier(n_estimator = n_estim, criterion = criterion, min_samples_split = min_samples, max_depth = max_depth)
	train = classifier.fit(X_train,Y_train)
	return train.predict(X_test)