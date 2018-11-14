import numpy as np
import matplotlib.pyplot as plt


def dc_plot(instances,array_1,title,x_label,y_label):
	plt.figure()
	plt.plot(instances,array_1, color = 'black', linestyle = '--', marker = 'X', markerfacecolor = 'red', markersize = 8)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

def all_classifiers(array_1,array_2,array_3,array_4,title,x_label,y_label):
	instances = np.arange(8)
	plt.figure()
	plt.plot(instances,array_1, color = 'black', linestyle = '--', marker = 'X', markerfacecolor = 'red', markersize = 8)
	plt.plot(instances,array_2, color = 'green', linestyle = '--', marker = 'X', markerfacecolor = 'yellow', markersize = 8)
	plt.plot(instances,array_3, color = 'blue', linestyle = '--', marker = 'X', markerfacecolor = 'black', markersize = 8)
	plt.plot(instances,array_4, color = 'red', linestyle = '--', marker = 'X', markerfacecolor = 'green', markersize = 8)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

def bar(dataframe,title,xlabel,ylabel):
	dataframe.plot.bar()
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()