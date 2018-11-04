import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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

def roc_curve_func(Y_test,predict,roc_color):

    fpr, tpr, _ = roc_curve(Y_test,predict)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr,tpr, color=roc_color,lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

def cross_val(X,Y,classifier,roc_color):

    accuracies_test = []
    k=1

    #Split the data with k folds for train and the rest to test 
    skf = StratifiedKFold(n_splits = 3, random_state = None, shuffle = True)
    for train_index, test_index in skf.split(X,Y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
        #Trainning the model
        model = classifier.fit(X_train,Y_train)
        #Prediction  
        Y_predict = model.predict(X_test)
        #Plot the model
        # plt_Prediction_Cross_Validation(Y_test,Y_predict,k)
        #Accuracy
        accuracy_test = accuracy_score(Y_test, Y_predict)
        #Put all accuricies in array to calculate mean accuracy
        accuracies_test.append(accuracy_test)
        #Count KFold
        k+=1
        #Calculate Confusion Matrix
        conf_matrix = confusion_matrix(Y_test, Y_predict)
        #Show Confusion Matrix
        plot_confusion_matrix(conf_matrix,classes)

        conf_matrix_sens_spec(Y_test,Y_predict)

        roc_curve_func(Y_test,predict,roc_color)

    return np.mean(accuracies_test),np.std(accuracies_test)




# data_bank = pd.read_csv('bank.csv')
# data_bank = preprocessData(data_bank)

# print data_bank.columns
# print data_bank['region']

# df = label_encoder.fit_transform(data_bank['married'])
# print df
# print data_bank['region'].values.reshape(-1,1)
# print len(data_bank['region_0'])

# X = data_bank.iloc[:,:data_bank.shape[1]-1]
# print X.cov().shape
# print data_bank.shape[1]-1
# Y = data_bank['pep']
# print Y

########## Exercicio1
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

# predict = naive_Bayes(X_train,X_test,Y_train)

# accuracy = accuracy_score(Y_test,predict)

# print 'Accuracy ', accuracy

# conf_matrix_sens_spec(Y_test,predict)

# roc_curve_func(Y_test,predict,'darkorange')

# plt.show()

########## Exercicio2

# predict, accuracy = Knn(X_train,X_test,Y_train,Y_test, 3)

# print 'Accuracy ',accuracy

# conf_matrix_sens_spec(Y_test,predict)

# roc_curve_func(Y_test,predict,'blue')

# plt.show()

########## Exercicio3

unbalanced_data = pd.read_csv('unbalanced.csv')
# unbalanced_data = preprocessData(unbalanced_data)

X = unbalanced_data.iloc[:,:-1]
# print X
Y = unbalanced_data['Outcome']
# print Y

classifier1 = GaussianNB()

classifier2 = KNeighborsClassifier(n_neighbors = 1)

classifier3 = KNeighborsClassifier(n_neighbors = 10)

classifier4 = KNeighborsClassifier(n_neighbors = 100)



# roc_curve_func(Y_test,predict,'darkorange')

# plt.show()













