import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracies = []
errors = []

def accuracy(prediction,Ytest):
	best_accuracy = 0
	#Accuracy
	accuracy = accuracy_score(Ytest, prediction)
	#Put all accuricies in array
	accuracies.append(accuracy)
	#Calculate the best accuracy
	best_accuracy = np.max(accuracies)
	#Calculate standart deviation
	std_accuracy = np.std(accuracies)

	print "Accuracy =",accuracies
	print "Best Accuracy =",best_accuracy
	print "Standart Deviation of Accuracy =",std_accuracy

	return accuracies

def error(prediction,Ytest):
	error_min = 0
	#Error
	error = np.mean(prediction != Ytest)
	#Put all errors in array
	errors.append(error)
	#Calculate the best accuracy
	error_min = np.min(errors)
	#Calculate standart deviation
	std_error = np.std(errors)

	print "Erro =",errors
	print "Min error =",error_min
	print "Standart Deviation of Error =",std_error

	return errors

def classi_report(prediction,Ytest):

	print classification_report(Ytest,prediction)

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
    plt.show()

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

