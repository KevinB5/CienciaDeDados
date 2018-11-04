import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

def plt_Prediction_Train_Test_Split(Y_test,Y_predict):

    #Plot the model
    plt.scatter(Y_test,Y_predict)
    plt.title('Train_Test_Train')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()

def plt_Prediction_Cross_Validation(Y_test,Y_predict,k):

    #Plot the model
    plt.scatter(Y_test,Y_predict)
    plt.title('Cross validation K = ' + str(k))
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()

def split_train_test(X,Y,classifier):
    #Split the data with 70% for train and the rest to test
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size = 0.8)
    #Trainning the model
    model = classifier.fit(X_train,Y_train)
    #Predicao  
    Y_predict = model.predict(X_test)
    #Plot the model
    # plt_Prediction_Train_Test_Split(Y_test,Y_predict)
    #Accuracy
    accuracy_test = accuracy_score(Y_test, Y_predict)
    #Calculate Confusion MAtrix
    conf_matrix =  confusion_matrix(Y_test, Y_predict)
    #show Confusion Matrix
    # plot_confusion_matrix(conf_matrix,classes)

    return accuracy_test

def cross_val(X,Y,classifier,splits):

    accuracies_test = []
    k=1

    #Split the data with 10 folds for train and the rest to test 
    skf = StratifiedKFold(n_splits = splits, random_state = None, shuffle = True)
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
        # plot_confusion_matrix(conf_matrix,classes)

    return np.mean(accuracies_test),np.std(accuracies_test)

# def cross_val_2(X,Y,classifier):

#     accuracies_test = []
#     k=1

#     #Split the data with 10 folds for train and the rest to test 
#     skf = StratifiedKFold(n_splits = 10, random_state = 0, shuffle = True)
#     #Prediction
#     est = cross_val_predict(classifier,X,Y, cv = skf)
#     #Prob Error rate
#     pErro = np.sum(est != Y)/(float)(len(est))

#     accuracy = cross_val_score(classifier,X,Y,cv = skf)

#     return np.mean(accuracy)


def exercise_1(X,Y,neighbors):

    #Instanciar o modelo de apredizagem com 3 neighbors (k = 3)
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    # Accuracy train_test_split
    print "Accuracy Test KNN splliting data with train_test =",split_train_test(X,Y,knn)
    # Accuracy StratifiedKFold
    print "Accuracy Test KNN splliting data with StratifiedKFold =",cross_val(X,Y,knn,splits)[0]
    # Standart desviation StratifiedKFold
    print "Standart desviation Test KNN splliting data with StratifiedKFold =",cross_val(X,Y,knn,splits)[1]

def exercise_2(X,Y):
    #Instanciar o modelo de apredizagem com Naive Bayes
    nb = GaussianNB()
    # Accuracy train_test_split
    print "Accuracy Test Naive Bayes splliting data with train_test =",split_train_test(X,Y,nb)
    # Accuracy StratifiedKFold
    print "Accuracy Test Naive Bayes splliting data with StratifiedKFold =",cross_val(X,Y,nb,splits)[0]
    # Standart desviation StratifiedKFold
    print "Standart desviation Test Naive Bayes splliting data with StratifiedKFold =",cross_val(X,Y,nb,splits)[1]

def exercise_3(X,Y,neighbors):

    accuricies_train_test = []
    accuricies_cross_val = []
    std_cross_val = []

    for i in neighbors:

        #Instanciar o modelo de apredizagem com 3 neighbors (k = 3)
        knn = KNeighborsClassifier(n_neighbors = i)
        accuricies_train_test.append(split_train_test(X,Y,knn))
        accuricies_cross_val.append(cross_val(X,Y,knn,splits)[0])
        std_cross_val.append(cross_val(X,Y,knn,splits)[1])

    # Accuracy train_test_split
    print "Accuracy Test KNN splliting data with train_test =",accuricies_train_test
    # Accuracy StratifiedKFold
    print "Accuracy Test KNN splliting data with StratifiedKFold =",accuricies_cross_val
    # Standart desviation StratifiedKFold
    print "Standart desviation Test KNN splliting data with StratifiedKFold =",std_cross_val

def exercise_4(X,Y):

    #Instanciar o modelo de apredizagem com Naive Bayes
    nb = GaussianNB()
    # Accuracy train_test_split
    print "Accuracy Test Naive Bayes splliting data with train_test =",split_train_test(X,Y,nb)
    # Accuracy StratifiedKFold
    print "Accuracy Test Naive Bayes splliting data with StratifiedKFold =",cross_val(X,Y,nb,splits)[0]
    # Standart desviation StratifiedKFold
    print "Standart desviation Test Naive Bayes splliting data with StratifiedKFold =",cross_val(X,Y,nb,splits)[1]


##########################################################################################

splits = 10

# #Leitura dos dados
# iris = pd.read_csv('iris.csv')
# #Instancias 
# X = iris.iloc[:,:4]
# #Classe correspondente aos dados
# Y = iris['class']
# #Classes existentes
# classes = np.unique(Y)

# exercise_1(X,Y,3)
# print ' '
# exercise_2(X,Y)

# idx_virg = np.argwhere(Y=='Iris-virginica')
# print idx_virg
# iris_setosa =  iris.loc[iris['class'] == 'Iris-setosa'].iloc[:,:4]
# iris_versicolor =  iris.loc[iris['class'] == 'Iris-versicolor'].iloc[:,:4]
# iris_virginica =  iris.loc[iris['class'] == 'Iris-virginica'].iloc[:,:4]
# print X[0:5]
# print Y

# plt.figure()
# ax = plt.axes(projection = '3d')
# ax.scatter3D(iris_setosa.iloc[:,0],iris_setosa.iloc[:,1],iris_setosa.iloc[:,2],'green')
# ax.scatter3D(iris_versicolor.iloc[:,0],iris_versicolor.iloc[:,1],iris_versicolor.iloc[:,2],'blue')
# ax.scatter3D(iris_virginica.iloc[:,0],iris_virginica.iloc[:,1],iris_virginica.iloc[:,2],'red')
# plt.show()


##########################################################################################


# neighbors = [1,5,10,15,50,100]

# glass = pd.read_csv('glass.csv')
# print glass.shape

# #Dados ate a 9 colunas
# X = glass.iloc[:,:9]
# #Classe correspondente aos dados
# Y = glass['Type']
# #Calsses existentes
# classes = np.unique(Y)

# print "Splits =",splits
# exercise_3(X,Y,[1,5,10,15,50,100])
# print ' '
# exercise_4(X,Y)