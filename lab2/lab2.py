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

def cross_val(X,Y,X_train,Y_train):
    cv_scores = []

    #Perfomance cross validation com 10-fold
    for i in range(3,len(X)-1):
        knn = KNeighborsClassifier(n_neighbors = i)
        scores = cross_val_score(knn,X_train,Y_train,cv = 10)
        cv_scores.append(scores.mean())

    return cv_scores


def exercise_1(X,Y,X_train,X_test,Y_train,Y_test,classes):

    #Instanciar o modelo de apredizagem com 3 neighbors (k = 3)
    knn = KNeighborsClassifier(n_neighbors = 3)
    #Treinar o modelo
    model = knn.fit(X_train,Y_train)
    #Predicao  
    Y_predict = model.predict(X_test)

    #a) Accuracy aproximadamente 93,3 %
    accuracy = accuracy_score(Y_test, Y_predict)
    print "Accuracy KNN -",accuracy

    c_validation = cross_val(X,Y,X_train,Y_train)

    # conf_matrix =  confusion_matrix(Y_test, Y_predict)

    # plot_confusion_matrix(conf_matrix,classes)

def exercise_2(X,Y,X_train,X_test,Y_train,Y_test,classes):

    nb = GaussianNB()
    model = nb.fit(X_train,Y_train)
    Y_predict = model.predict(X_test)

    #a)
    accuracy = accuracy_score(Y_test, Y_predict)
    print "Accuracy NB -",accuracy

    validation = cross_val_score(nb,X,Y,cv=10)
    print "Validation NB -",validation

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
#Split dos dados com 70% para train e os restantes para teste
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size = 0.7, stratify = Y)


exercise_1(X,Y,X_train,X_test,Y_train,Y_test,classes)
# print ' '
# exercise_2(X,Y,X_train,X_test,Y_train,Y_test,classes)


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