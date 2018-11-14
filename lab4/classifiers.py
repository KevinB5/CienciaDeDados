from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

def dc(xTrain,yTrain,xTest,crite,depth = None,split = 2,leaf = None):
	#Inicialize classifier and put default values
	clf = DecisionTreeClassifier(criterion = crite,max_depth = depth,min_samples_split = split,max_leaf_nodes = leaf,random_state = 0)
	#Training data
	clf.fit(xTrain,yTrain)
	#Return what the classfier predict in relation the training data
	return clf.predict(xTest)