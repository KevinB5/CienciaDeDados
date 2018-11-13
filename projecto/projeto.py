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

#Leitura dos dados Digital Colposcopies
#green = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/Quality Assessment-Digital Colposcopy/green.csv')
#hinselmann = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/Quality Assessment-Digital Colposcopy/hinselmann.csv')
#schiller = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/Quality Assessment-Digital Colposcopy/schiller.csv')

#Verificar se o formato é utilizável 
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
aps_test = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/aps_failure/aps_failure_test_set.csv')
aps_training = pd.read_csv('C:/Users/kevin\Documents/GitHub/CienciaDeDados/projecto/aps_failure/aps_failure_training_set.csv')

def delete_trash_columns(dataset,percentage):
    for column in dataset.columns:
        if sum(dataset[column].isnull())/float(len(dataset[column].index)) > percentage:
            dataset.drop([column], axis = 1, inplace = True)

#aps_test = aps_test.replace('neg',np.nan)
#aps_test = aps_test.replace('pos',np.nan)
aps_test = aps_test.replace('na',np.nan)
delete_trash_columns(aps_test,0.45)

cleanup = {"neg": 0, "pos": 1}
aps_test.replace(cleanup, inplace = True)

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

#print(preprocessData(aps_test).head())

col_mean = np.nanmean(aps_test, axis=0,dtype='float64')
print(col_mean)

def replace_missing_values_mean(dataset):
    for column in dataset.columns:
        print(column)
        #print('oi')
        # print(column)
            
        #print(np.nanmean(dataset[column],'all'))
        #column_mean = np.nanmean(dataset[column])
        #print(column_mean)
        #dataset[column] = dataset[column].replace(np.nan,column_mean)

#aps_test = replace_missing_values_mean(aps_test)

#print(aps_test)





