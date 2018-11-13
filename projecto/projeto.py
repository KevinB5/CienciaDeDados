import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb
import os


dir_name =  os.path.dirname(__file__)
path_test = os.path.normpath('aps_failure/aps_failure_test_set.csv')
path_train = os.path.normpath('aps_failure/aps_failure_training_set.csv')
# print os.path.dirname(path)
path_file1 = os.path.join(os.getcwd(),path_test)
path_file2 = os.path.join(os.getcwd(),path_train)

aps_test = pd.read_csv(path_file1)
aps_training = pd.read_csv(path_file2)

#Leitura dos dados Digital Colposcopies
#green = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/Quality Assessment-Digital Colposcopy/green.csv')
#hinselmann = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/Quality Assessment-Digital Colposcopy/hinselmann.csv')
#schiller = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/Quality Assessment-Digital Colposcopy/schiller.csv')

#Verificar se o formato e utilizavel 
#green.head()
#hinselmann.head()
#schiller.head()

#Remover os dados vazios
#print(green.columns.values)

##green = green.replace(0.0,np.nan)
##green = green.dropna()
#green = green.replace(0.0,green.mean())
#print(green.head())

##hinselmann = hinselmann.replace(0.0,np.nan)
##hinselmann = hinselmann.dropna()
#hinselmann = hinselmann.replace(0.0,hinselmann.mean())
#print(hinselmann.head())

##schiller = schiller.replace(0.0,np.nan)
##schiller = schiller.dropna()
#schiller = schiller.replace(0.0,schiller.mean())
#print(schiller.head())

#Leitura dos dados APS failure at Scania trucks


def preprocessData(df):
    label_encoder = preprocessing.LabelEncoder()
    dummy_encoder = preprocessing.OneHotEncoder()
    pdf = pd.DataFrame()
    for att in df.columns:
        if df[att].dtype == np.float64 or df[att].dtype == np.int64:
            pdf = pd.concat([pdf, df[att]], axis=1)

        elif len(np.unique(df[att])) == 2:
                df[att] = label_encoder.fit_transform(df[att])
                pdf = pd.concat([pdf,df[att]], axis=1)
        else:
            df[att] = label_encoder.fit_transform(df[att])
            # Fitting One Hot Encoding on train data
            temp = dummy_encoder.fit_transform(df[att].values.reshape(-1,1)).toarray()
            # Changing encoded features into a dataframe with new column names
            temp = pd.DataFrame(temp,
                                columns=[(att + "_" + str(i)) for i in df[att].value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            pdf = pd.concat([pdf, temp], axis=1)
    return pdf

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

def separate_data(df,Y,major):

    # df_majority = new DataFrame()
    # df_minority = new DataFrame()

    if(major == 1):
        df_majority = df[Y==1]
        df_minority = df[Y==0]
    else:
        df_majority = df[Y==0]
        df_minority = df[Y==1]

    return df_majority,df_minority
        

def balance_data(df,Y,major,replace):

    df_majority, df_minority = separate_data(df,Y,major)

    # print df_majority

    if(replace==True):
        df_minority = resample(df_minority,replace=replace,n_samples=len(df_majority),random_state=123)
    else:
        df_majority = resample(df_majority,replace=replace,n_samples=len(df_minority),random_state=123)

    df = pd.concat([df_minority, df_majority])
    df = df.reset_index()
    df = df.drop(['index'],axis=1)

    return df



X = aps_test.iloc[:,1:]
print X

Y = aps_test.iloc[:,0]
print Y

aps_replaced = X.replace('na',np.nan)
delete_trash_columns(aps_replaced,0.45)

#print(preprocessData(aps_test).head())

aps_test = replace_missing_values_mean(aps_replaced)

print(aps_test)





