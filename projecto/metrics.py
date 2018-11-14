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

	# print "Accuracy =",accuracies
	# print "Best Accuracy =",best_accuracy
	# print "Standart Deviation of Accuracy =",std_accuracy

	return accuracy

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

def roc_curve_func(Y_test,predict,roc_color,title):

    fpr, tpr, _ = roc_curve(Y_test,predict)
    roc_auc = auc(fpr,tpr)

    # plt.figure()
    plt.plot(fpr,tpr, color=roc_color,lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")