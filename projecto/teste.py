import os
import numpy as np
import pandas as pd


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

df = X
print pd.concat((Y,X),axis=1)

# print aps_test