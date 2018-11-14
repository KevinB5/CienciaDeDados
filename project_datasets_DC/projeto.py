import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from strategies import *
from metrics import *
from pre_processing import *
from plots import *


dir_name =  os.path.dirname(__file__)
path_green= os.path.normpath('datasets/digital_colposcopy/green.csv')
path_hinselmann = os.path.normpath('datasets/digital_colposcopy/hinselmann.csv')
path_schiller = os.path.normpath('datasets/digital_colposcopy/schiller.csv')
# print os.path.dirname(path)
path_file1 = os.path.join(os.getcwd(),path_green)
path_file2 = os.path.join(os.getcwd(),path_hinselmann)
path_file3 = os.path.join(os.getcwd(),path_schiller)

green = pd.read_csv(path_file1)
hinselmann = pd.read_csv(path_file2)
schiller = pd.read_csv(path_file3)

#Faz sentido juntar todos porque o que interssa saber a se a pessoa esta bem ou nao, em vez de saber qual o matodo utilizado
df = pd.concat([green,hinselmann,schiller])
df = df.reset_index()
df = df.drop(['index'],axis=1)

X = df.iloc[:,:62]

Y = df.iloc[:,62:]

# delete_trash_columns(df_x,0.5)

# df_x = replace_missing_values_mean(df_x)

for i in range(7):
    df_y = Y.iloc[:,i]
    print df_y.describe()
    df_accuracies,df_medias,df_std = cross_val(X,df_y,3)
    bar(df_accuracies,'Target Variables with Accuracies','KFolds','Accuracies')
    # bar(df_medias,'Target Variables with Mean','KFolds','Mean each Accuracy')
    # bar(df_std,'Target Variables with Std','KFolds','Std each Accuracy')