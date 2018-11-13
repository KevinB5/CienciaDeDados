import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

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

def plot_confusion_matrix(cnf_matrix, classesNames, normalize=False, cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if normalize:
        soma = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / soma
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix, without normalization'

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classesNames))
    plt.xticks(tick_marks, classesNames, rotation=45)
    plt.yticks(tick_marks, classesNames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def naive_Bayes(X_train,X_test,Y_test):

    gnb = GaussianNB()

    model = gnb.fit(X_train,Y_train)

    predict = model.predict(X_test)

    return predict

def conf_matrix_sens_spec(Y_test,predict):

    conf_matrix = confusion_matrix(Y_test,predict)
    tn, fp, fn, tp = confusion_matrix(Y_test,predict).ravel()
    print 'Confusion Matrix '
    print conf_matrix

    sensitivity = tp*1./(tp+fn)
    print 'Sensitivity ',sensitivity

    specificity = tn*1./(tn+fp)
    print 'Specificity ',specificity

def roc_curve_func(Y_test,predict,roc_color,title):

    fpr, tpr, _ = roc_curve(Y_test,predict)
    roc_auc = auc(fpr,tpr)

    # plt.figure()
    plt.plot(fpr,tpr, color=roc_color,lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

def cross_val(X,Y,classifier,roc_color,title):

    accuracies_test = []
    accuracies_train = []
    k=0
    print "------------------ "+title+" --------------------"
    #Split the data with k folds for train and the rest to test 
    skf = StratifiedKFold(n_splits = 3, random_state = None, shuffle = True)
    for train_index, test_index in skf.split(X,Y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
        #Trainning the model
        model = classifier.fit(X_train,Y_train)
        #Prediction  
        Y_predict = model.predict(X_test)
        #Prediction
        YX_predict = model.predict(X_train)
        #Plot the model
        # plt_Prediction_Cross_Validation(Y_test,Y_predict,k)
        #Accuracy
        accuracy_test = accuracy_score(Y_test, Y_predict)
        #Put all accuricies in array to calculate mean accuracy
        accuracies_test.append(accuracy_test)

        accuracy_train = accuracy_score(Y_train, YX_predict)
        #Put all accuricies in array to calculate mean accuracy
        accuracies_train.append(accuracy_train)
        
        
        #Calculate Confusion Matrix
        # conf_matrix = confusion_matrix(Y_test, Y_predict)
        #Show Confusion Matrix
        # plot_confusion_matrix(conf_matrix,classes)

        conf_matrix_sens_spec(Y_test,Y_predict)

        roc_curve_func(Y_test,Y_predict,roc_color[k],title)
        #Count KFold    
        k+=1

    # print "Accuracy Test -> ",accuracies_test
        print classification_report(Y_test,Y_predict,target_names=['class0','class1'])



    return accuracies_test,np.std(accuracies_test),accuracies_train,np.std(accuracies_train)

def cross_val2(X,Y,classifier,roc_color,title):

    accuracies_test = []
    accuracies_train = []
    k=0
    print "------------------ "+title+" --------------------"
    #Split the data with k folds for train and the rest to test 
    skf = StratifiedKFold(n_splits = 3, random_state = None, shuffle = True)
    for train_index, test_index in skf.split(X,Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        #Trainning the model
        model = classifier.fit(X_train,Y_train)
        #Prediction  
        Y_predict = model.predict(X_test)
        #Prediction
        YX_predict = model.predict(X_train)
        #Plot the model
        # plt_Prediction_Cross_Validation(Y_test,Y_predict,k)
        #Accuracy
        accuracy_test = accuracy_score(Y_test, Y_predict)
        #Put all accuricies in array to calculate mean accuracy
        accuracies_test.append(accuracy_test)

        accuracy_train = accuracy_score(Y_train, YX_predict)
        #Put all accuricies in array to calculate mean accuracy
        accuracies_train.append(accuracy_train)
        
        
        #Calculate Confusion Matrix
        # conf_matrix = confusion_matrix(Y_test, Y_predict)
        #Show Confusion Matrix
        # plot_confusion_matrix(conf_matrix,classes)

        conf_matrix_sens_spec(Y_test,Y_predict)

        roc_curve_func(Y_test,Y_predict,roc_color[k],title)
        #Count KFold    
        k+=1

    # print "Accuracy Test -> ",accuracies_test
        print classification_report(Y_test,Y_predict,target_names=['class0','class1'])



    return accuracies_test,np.std(accuracies_test),accuracies_train,np.std(accuracies_train)

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

def compare_classifiers(X,Y,roc_colors,classifiers,classifier_names):

    plt.figure(figsize = (6*(len(classifiers)),5*(len(classifiers))))
    k = 1
    for i in range(len(classifiers)):
        
        plt.subplot((110*len(classifiers))+i+k)
        crossval_class = cross_val2(X,Y,classifiers[i],roc_colors,classifier_names[i])
        plt.subplot((110*len(classifiers))+i+k+1)
        plt.boxplot([crossval_class[0],crossval_class[2]],positions=[0,1],widths=0.6)
        k=k+1
        print "Accuracy Test"
        print crossval_class[0], crossval_class[1]
        print "Accuracy Train"
        print crossval_class[2], crossval_class[3]

    plt.show()


# data_bank = pd.read_csv('bank.csv')
# data_bank = preprocessData(data_bank)

# print data_bank.columns
# print data_bank['region']

# df = label_encoder.fit_transform(data_bank['married'])
# print df
# print data_bank['region'].values.reshape(-1,1)
# print len(data_bank['region_0'])

# X = data_bank.iloc[:,:data_bank.shape[1]-1]
# print X
# print X.cov().shape
# print data_bank.shape[1]-1
# Y = data_bank['pep']
# print Y

# pep0 =  X.loc[Y == 0,['age','income','children']]
# # print pep0
# pep1 =  X.loc[Y == 1,['age','income','children']]

# plt.figure(figsize = (12,10))
# plt.subplot(221)
# plt.scatter(pep0.iloc[:,0],pep0.iloc[:,1],color='green')
# plt.scatter(pep1.iloc[:,0],pep1.iloc[:,1],color='blue')
# plt.title('Bank DataBase')
# plt.xlabel("Age")
# plt.ylabel("Income")
# plt.show()
# plt.figure()
# ax = plt.axes(projection = '3d')
# ax.scatter3D(pep0.iloc[:,0],pep0.iloc[:,1],pep0.iloc[:,2],color='green')
# ax.scatter3D(pep1.iloc[:,0],pep1.iloc[:,1],pep1.iloc[:,2],color='blue')

# plt.show()

########## Exercicio1

#Sensitivity: True Positive Rate 
#the true positive rate, also called sensitivity,
#is the fraction of correct predictions with respect to all points in the positive class,
#that is, it is simply the recall for the positive class
#   TPR = recallP = TP/TP+FN = TP/n1

#Specificity: True Negative Rate 
#The true negative rate, also called specificity, is simply the recall for the negative class:
#   TNR = specificity = recallN = TN/FP+TN = TN/n2 # where n2 is the size of the negative class.

#False Negative Rate
#The false negative rate is defined as
#   FNR=FN/TP+FN = FN/n1 = 1-sensitivity

#False Positive Rate 
#The false positive is defined as 
#   FPR=FP/FP+TN = FP/n2 = 1-specificity

# classifier1 = GaussianNB()
# roc_colors = ['darkorange', 'blue', 'green']

# cross_val(X,Y,classifier1,roc_colors)
# plt.show()

##############################################################################

########## Exercicio2

# pep0 =  X.loc[Y == 0,['age','income','children']]
# # print pep0
# pep1 =  X.loc[Y == 1,['age','income','children']]

# plt.figure(figsize = (12,10))
# plt.subplot(221)
# plt.scatter(pep0.iloc[:,0],pep0.iloc[:,1],color='green')
# plt.scatter(pep1.iloc[:,0],pep1.iloc[:,1],color='blue')
# plt.title('Bank DataBase')
# plt.xlabel("Age")
# plt.ylabel("Income")

# classifier1 = GaussianNB()
# classifier2 = KNeighborsClassifier(n_neighbors = 3)

# roc_colors = ['darkorange', 'blue', 'green']

# plt.subplot(222)
# cross_val(X,Y,classifier1,roc_colors,'Naive_Bayes')
# # plt.show()
# plt.subplot(223)
# cross_val(X,Y,classifier2,roc_colors,'KNeighborsClassifier')

# plt.show()

##############################################################################

########## Exercicio3

unbalanced_data = pd.read_csv('unbalanced.csv')
unbalanced_data = preprocessData(unbalanced_data)
# print unbalanced_data.isnan()

X = unbalanced_data.iloc[:,:-1]
# print X.iloc[:24]
Y = unbalanced_data['Outcome']
# # print Y
# print Y.value_counts() #### Ver quantas instancias de cada classe existem

# balanced_data = balance_data(unbalanced_data,unbalanced_data['Outcome'],1,True)
# print balanced_data.isnan()
# print np.any(np.isnan(balanced_data))
# print np.all(np.isfinite(balanced_data))

# print balanced_data

# X_balanced = balanced_data.iloc[:,:-1]
# # print X_balanced
# # print X_balanced == X.iloc[:,:24]
# # print len(X.iloc[:24])
# Y_balanced = balanced_data['Outcome']
# print Y_balanced.value_counts()


X_balanced,Y_balanced = balance_SMOTE(X,Y)




classifier_NB = GaussianNB()

classifier_Knn1 = KNeighborsClassifier(n_neighbors = 1)

classifier_Knn3 = KNeighborsClassifier(n_neighbors = 3)

classifier_Knn10 = KNeighborsClassifier(n_neighbors = 10)

classifier4_Knn100 = KNeighborsClassifier(n_neighbors = 100)

# outcome0 =  X.loc[Y == 0,['age','income','children']]
# # # print pep0
# outcome0 =  X.loc[Y == 1,['age','income','children']]

# plt.figure(figsize = (12,10))
# plt.subplot(221)
# plt.scatter(outcome0.iloc[:,0],outcome0.iloc[:,1],color='green')
# plt.scatter(outcome0.iloc[:,0],outcome0.iloc[:,1],color='blue')
# plt.title('Bank DataBase')
# plt.xlabel("Age")
# plt.ylabel("Income")

roc_colors = ['darkorange', 'blue', 'green']
classifiers_names = ['Naive_Bayes','1 - NeighborsClassifier','3 - NeighborsClassifier', '10 - NeighborsClassifier', '100 - NeighborsClassifier']

compare_classifiers(X_balanced,Y_balanced,roc_colors,[classifier_NB,classifier_Knn3, classifier_Knn10],classifiers_names)

# plt.subplot(221)
# crossval_NB = cross_val(X_balanced,Y_balanced,classifier_NB,roc_colors,'Naive_Bayes')
# plt.subplot(222)
# plt.boxplot([crossval_NB[0],crossval_NB[2]])

# print "Accuracy Test"
# print crossval_NB[0], crossval_NB[1]
# print "Accuracy Train"
# print crossval_NB[2], crossval_NB[3]
# # plt.show()
# plt.subplot(223)
# crossval_Knn = cross_val(X_balanced,Y_balanced,classifier_Knn3,roc_colors,'KNeighborsClassifier')
# plt.subplot(224)
# plt.boxplot([crossval_Knn[0],crossval_Knn[2]])

# print "Accuracy Test"
# print crossval_Knn[0], crossval_Knn[1]
# print "Accuracy Train"
# print crossval_Knn[2], crossval_Knn[3]

# plt.show()