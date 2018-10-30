import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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

def split_train_test(X,Y):
    #Split dos dados com 70% para train e os restantes para teste
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size = 0.7, stratify = Y)

    return X_train,X_test,Y_train,Y_test


def cross_val(X,Y):

    #Split dos dados com 10 folds para train e os restantes para teste
    skf = StratifiedKFold(n_splits = 10, random_state = True)
    for train_index, test_index in skf.split(X,Y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]

    return X_train,X_test,Y_train,Y_test


def exercise_1(X,Y,classes):

    # Split data with train_test_split
    X_train,X_test,Y_train,Y_test = split_train_test(X,Y)
    # Split data with StratifiedKFold
    # X_train,X_test,Y_train,Y_test = cross_val(X,Y)

    #Instanciar o modelo de apredizagem com 3 neighbors (k = 3)
    knn = KNeighborsClassifier(n_neighbors = 3)
    #Treinar o modelo
    model = knn.fit(X_train,Y_train)
    #Predicao  
    Y_predict = model.predict(X_test)

    #a) Accuracy com train_test_split varia entre 93,3% e 100%
    accuracy_train_test = accuracy_score(Y_test, Y_predict)
    print "Accuracy KNN splliting data with train_test =",accuracy_train_test

    #a) Accuracy com StratifiedKFold varia entre 
    # accuracy_StratifiedKFold = accuracy_score(Y_test, Y_predict)
    # print "Accuracy KNN splliting data with StratifiedKFold =",accuracy_StratifiedKFold

    # conf_matrix =  confusion_matrix(Y_test, Y_predict)

    # plot_confusion_matrix(conf_matrix,classes)

def exercise_2(X,Y,classes):

    # Split data with train_test_split
    X_train,X_test,Y_train,Y_test = split_train_test(X,Y)
    # Split data with StratifiedKFold
    # X_train,X_test,Y_train,Y_test = cross_val(X,Y)

    nb = GaussianNB()

    model = nb.fit(X_train,Y_train)

    Y_predict = model.predict(X_test)

    #a) Accuracy com train_test_split varia entre 93,3% e 100%
    accuracy_train_test = accuracy_score(Y_test, Y_predict)
    print "Accuracy Naive_Bayes splliting data with train_test =",accuracy_train_test

    #a) Accuracy com StratifiedKFold varia entre 
    # accuracy_StratifiedKFold = accuracy_score(Y_test, Y_predict)
    # print "Accuracy Naive_Bayes splliting data with StratifiedKFold =",accuracy_StratifiedKFold

    # conf_matrix = confusion_matrix(Y_test, Y_predict)

    # plot_confusion_matrix(conf_matrix,classes)

def exercise_3(neighbors,X_train,X_test,Y_train,Y_test,classes):

    #a,b)
    accuracies = []

    for i in neighbors:
        knn = KNeighborsClassifier(n_neighbors = i)
        model = knn.fit(X_train,Y_train)
        Y_predict = model.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_predict)
        accuracies.append(accuracy)

    return accuracies

def exercise_4(X_train,X_test,Y_train,Y_test,classes):

    knn = GaussianNB()
    model = knn.fit(X_train,Y_train)
    Y_predict = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_predict)

    print accuracy


##########################################################################################

#Leitura dos dados
iris = pd.read_csv('iris.csv')
#Instancias 
X = iris.iloc[:,:4]
#Classe correspondente aos dados
Y = iris['class']
#Classes existentes
classes = np.unique(Y)


exercise_1(X,Y,classes)
print ' '
exercise_2(X,Y,classes)


##########################################################################################


# neighbors = [1,5,10,15,50,100]

# glass = pd.read_csv('glass.csv')

# #Dados ate a 9 colunas
# X = glass.iloc[:,:9]
# #Classe correspondente aos dados
# Y = glass['Type']
# #Calsses existentes
# classes = np.unique(Y)

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

# print exercise_3(neighbors,X_train,X_test,Y_train,Y_test,classes)
# print exercise_4(X_train,X_test,Y_train,Y_test,classes)