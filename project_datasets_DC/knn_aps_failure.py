import pandas as pd
from classifiers import *
from pre_processing import *
from strategies import *

path_file = 'datasets/aps_failure/aps_failure_training_set.csv'

train = pd.read_csv(path_file,engine='python',na_values='na',keep_default_na=True)

print train.head()
print train.describe()
print train["class"].value_counts()