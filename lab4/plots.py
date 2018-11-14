import numpy as np
import matplotlib.pyplot as plt


def dc_plot(instances,array_1,title,x_label,y_label):
	plt.figure()
	plt.plot(instances,array_1, color = 'black', linestyle = '--', marker = 'X', markerfacecolor = 'red', markersize = 8)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()