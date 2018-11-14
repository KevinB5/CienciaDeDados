import os
import numpy as np
import pandas as pd
from scipy.stats import zscore


dir_name =  os.path.dirname(__file__)
path_test = os.path.normpath('aps_failure/aps_failure_test_set.csv')
path_train = os.path.normpath('aps_failure/aps_failure_training_set.csv')
# print os.path.dirname(path)
path_file1 = os.path.join(os.getcwd(),path_test)
path_file2 = os.path.join(os.getcwd(),path_train)

aps_test = pd.read_csv(path_file1)
aps_training = pd.read_csv(path_file2)

X = aps_test.iloc[:,1:]
Y = aps_test.iloc[:,0]
# print X
# print Y
X = X.replace('na',np.nan)
# X = X.apply(np.float64)

df = X
# print pd.concat((Y,X),axis=1)

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

# X = X.replace('na',np.nan)
delete_trash_columns(X,0.5)

X = replace_missing_values_mean(X)
# X = X.apply(np.float64)
print X.iloc[1:,:].apply(zscore)

# print aps_test