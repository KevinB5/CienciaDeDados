import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE


dir_name =  os.path.dirname(__file__)
path_test = os.path.normpath('aps_failure/aps_failure_test_set.csv')
path_train = os.path.normpath('aps_failure/aps_failure_training_set.csv')
# print os.path.dirname(path)
path_file1 = os.path.join(os.getcwd(),path_test)
path_file2 = os.path.join(os.getcwd(),path_train)

aps_test = pd.read_csv(path_file1)
aps_training = pd.read_csv(path_file2)

#print(aps_test.head())
#print(aps_training.head())

#print(aps_test.shape)
#print(aps_training.shape)

#print(aps_test['class'].unique())
#print(aps_training['class'].unique())

aps_test = aps_test.replace('na',np.nan)
aps_training = aps_training.replace('na',np.nan)

#print(aps_test.isna().sum())
#print(aps_training.isna().sum())


def delete_trash_columns(dataset,percentage):
    for column in dataset.columns:
        if sum(dataset[column].isnull())/float(len(dataset[column].index)) > percentage:
            dataset.drop([column], axis = 1, inplace = True)

def replace_missing_values_mean(dataset):
    col_mean = np.nanmean(dataset, axis=0,dtype='float64')
    i = 0
    for column in dataset.columns:
        dataset[column] = dataset[column].fillna(col_mean[i]) 
        i=i+1   
    return dataset
        

def balance_smote(x, y):
    sm = SMOTE(random_state=2)
    x_res, y_res = sm.fit_sample(x, y.ravel())
    return x_res, y_res


aps_test = aps_test.replace('na',np.nan)
delete_trash_columns(aps_test,0.45)

aps_training = aps_training.replace('na',np.nan)
delete_trash_columns(aps_training,0.45)

aps_test_values = aps_test.iloc[:,1:]
aps_test_classes = aps_test.iloc[:,0]
    
aps_training_values = aps_training.iloc[:,1:]
aps_training_classes = aps_training.iloc[:,0]

aps_test_values = replace_missing_values_mean(aps_test_values)
aps_training_values = replace_missing_values_mean(aps_training_values)

#print(aps_test_classes.value_counts())
#print(aps_training_classes.value_counts())

def bar(dataframe,title,xlabel,ylabel):
	dataframe.plot.bar()
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()
    
bar(aps_training,'teste','x','y')

aps_training_values,aps_training_classes = balance_smote(aps_training_values,aps_training_classes) 
print('Resampled dataset shape {}'.format(Counter(aps_training_classes)))





