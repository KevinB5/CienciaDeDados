import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import numpy as np

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


def naive_Bayes(X_train,X_test,Y_test):

    gnb = GaussianNB()

    model = gnb.fit(X_train,Y_train)

    predict = model.predict(X_test)

    return predict

def conf_matrix_sens_spec(Y_test,predict):

    conf_matrix = confusion_matrix(Y_test,predict)
    tn, fp, fn, tp = confusion_matrix(Y_test,predict).ravel()
    print 'Confusion Matrix ', conf_matrix

    sensitivity = tp*1./(tp+fn)
    print 'Sensitivity ',sensitivity

    specificity = tn*1./(tn+fp)
    print 'Specificity ',specificity

def roc_curve_func(Y_test,predict):

    fpr, tpr, _ = roc_curve(Y_test,predict)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr,tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()



data_bank = pd.read_csv('bank.csv')
data_bank = preprocessData(data_bank)

print data_bank.columns
# print data_bank['region']

# df = label_encoder.fit_transform(data_bank['married'])
# print df
# print data_bank['region'].values.reshape(-1,1)
# print len(data_bank['region_0'])

X = data_bank.iloc[:,:data_bank.shape[1]-1]

# print data_bank.shape[1]-1
Y = data_bank['pep']
# print Y

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

predict = naive_Bayes(X_train,X_test,Y_train)

accuracy = accuracy_score(Y_test,predict)

print 'Accuracy ', accuracy

conf_matrix_sens_spec(Y_test,predict)

roc_curve_func(Y_test,predict)











