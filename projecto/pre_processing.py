import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore

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

def z_score(X):

	return X.apply(zscore)

def min_max(X):

	min_max = preprocessing.MinMaxScaler()

	return min_max.fit_transform(X)

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

def balance_SMOTE(X,Y):

    sm = SMOTE(random_state=12, ratio = 1.0)
    x_train_res, y_train_res = sm.fit_sample(X,Y)

    return x_train_res, y_train_res

def pre_processing(X,Y,major,replace,percentage=0.5,dummies=1,delete_columns=1,replace_na=1,normalize=0,balance_data=0,balance_SMOTE=0):

	X_aux = X
	y_aux = Y

	X_aux = X_aux.replace('na',np.nan)
	X_aux = X.apply(np.float64)

	if(dummies):

		X_aux = preprocessData(X_aux)

	if(delete_columns):

		delete_trash_columns(X_aux)

	if(replace_na):

		X_aux = replace_missing_values_mean(X_aux)

	if(normalize):
		X_aux = z_score(X_aux)
	else:
		X_aux = min_max(X_aux)

	if(balance_data):
 
		df_aux = pd.concat((y_aux,X_aux),axis=1)
		df_aux = balance_data(df_aux,y_aux,major,replace)
		#APS
		X_aux = df_aux.iloc[:,1:]
		y_aux = df_aux.iloc[:,0]

	if(balance_SMOTE):

		X_aux, y_aux = balance_SMOTE(X_aux,y_aux)

	return X_aux, y_aux