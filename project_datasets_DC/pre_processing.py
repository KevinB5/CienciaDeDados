import pandas as pd
import numpy as np
from sklearn import preprocessing

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