import numpy as np

#Calculate the best accuracy
best_accuracy_test = 0
# best_accuracy_train = 0
accuracies_test = []
# accuracies_train = []
#Trainning the model
model = classifier.fit(X_train,Y_train)	
#Prediction Test
Y_predict = classifier.predict(X_test)
#Prediction Train
# YX_predict = classifier.predict(X_train)
#Plot the model
# plt_Prediction_Cross_Validation(Y_test,Y_predict,k)
#Accuracy Test
accuracy_test = accuracy_score(Y_test, Y_predict)
#Put all accuricies in array to calculate mean accuracy
accuracies_test.append(accuracy_test)
# #Accuracy Train
# accuracy_train = accuracy_score(Y_train, YX_predict)
# #Put all accuricies in array to calculate mean accuracy
# accuracies_train.append(accuracy_train)
#Best Accuracy Score
# if(best_accuracy_test < accuracy_test or best_accuracy_train < accuracy_train): # Se quiser a accuracy do train para ver se esta ou nao em overfiting
if(best_accuracy_test < accuracy_test):
	best_accuracy_test = accuracy_test
	# best_accuracy_train = accuracy_train
#Count KFold
k+=1
#Calculate Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_predict)

#Show Confusion Matrix
# plot_confusion_matrix(conf_matrix,classes)
# print t_test(accuracies_train,accuracies_test)[1]