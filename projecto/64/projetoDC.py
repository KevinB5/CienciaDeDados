import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


dir_name =  os.path.dirname(__file__)
green_path = os.path.normpath('Quality Assessment-Digital Colposcopy/green.csv')
hinselmann_path = os.path.normpath('Quality Assessment-Digital Colposcopy/hinselmann.csv')
schiller_path = os.path.normpath('Quality Assessment-Digital Colposcopy/schiller.csv')

path_file1 = os.path.join(os.getcwd(),green_path)
path_file2 = os.path.join(os.getcwd(),hinselmann_path)
path_file3 = os.path.join(os.getcwd(),schiller_path)

green = pd.read_csv(path_file1)
hinselmann = pd.read_csv(path_file2)
schiller = pd.read_csv(path_file3)

#print(green.head())
#print(hinselmann.head())
#print(schiller.head())

#print(green.shape)
#print(hinselmann.shape)
#print(schiller.shape)

print(green.isna().sum())
print(hinselmann.isna().sum())
print(schiller.isna().sum())

def replace_missing_values_mean(dataset):
    col_mean = np.nanmean(dataset, axis=0,dtype='float64')
    i = 0
    for column in dataset.columns:
        dataset[column] = dataset[column].fillna(col_mean[i]) 
        i=i+1   
    return dataset

def preprocess(df,class_name):
    df_aux = df

    original_columns = df_aux.columns

    min_max_scaler = preprocessing.MinMaxScaler()
    df_aux = min_max_scaler.fit_transform(df_aux)
    
    df_aux = pd.DataFrame(df_aux, columns = original_columns)

    x = np.array(df_aux.loc[:, df_aux.columns != class_name])
    y = np.array(df_aux.loc[:, df_aux.columns == class_name])

    
    x, y = balance_smote(x, y, print_details = 1)
    rus = RandomUnderSampler(random_state=42)
    x, y = rus.fit_resample(x, y)

    return x,y        

def balance_smote(x, y):

    sm = SMOTE(random_state=2)

    x_res, y_res = sm.fit_sample(x, y.ravel())
    
    return x_res, y_res





