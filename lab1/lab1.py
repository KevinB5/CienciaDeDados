import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing


#1a
data = pd.read_csv('algae.csv')
season = data['season']
ph = data['pH']
Ammonium = data['Ammonium']
# print data.shape
# print data.iloc[0,:][3]

# print len([x for x in data.iloc[0,:] if type(x) == np.float64])

# print data.head()
# print data.describe()
# print data['season'].describe()
# print season.head()
# print season.describe()

seasons = np.unique(season)
# spring = data[season == seasons[0]]
# print len(spring)
# summer = data[season == seasons[1]]
# print len(summer)
# autumn = data[season == seasons[2]]
# print len(autumn)
# winter = data[season == seasons[3]]
# print len(winter)

# season_bins = np.arange(0,200,10)
# data['season'].plot(kind='hist', bins = season_bins, legend = True)

# print ph.dtype
# print ph.describe()
# print [x for x in np.unique(ph) if math.isnan(x) == False]
# print len([x for x in np.unique(ph) if math.isnan(x) == True])


########### E) OUTLIERS
# ###Retirar valores nan
# ph = ph[~np.isnan(ph)]
# # ph = np.sort(ph)
# # print ph[:50]
# Q1 = ph.quantile(0.25)
# Q3 = ph.quantile(0.75)
# IQR = Q3 - Q1

# print "Q1", Q1
# print "Q3", Q3
# print "IQR", IQR

# outliers = ph[(ph < (Q1 - 1.5 * IQR)) | (ph > (Q3 + 1.5 * IQR)) == True]

# # print ph
# plt.boxplot(ph)
# plt.show()	


########### pH in Winter

# season_ph = data.loc[:,['season','pH']]
# # print season_ph

# # print season_ph['season'] == seasons[0]
# pH_Winter = season_ph[season_ph['season'] == seasons[0]]
# print pH_Winter['pH']

# fig = plt.figure(figsize=(17,10))
# ph.hist()
# pH_Winter['pH'].hist()
# plt.title('ph Winter')
# plt.xlabel('valores de pH')
# plt.ylabel('freq')
# plt.show()

########## pH and Ammonium

### Nao tem a mesma escala, o que pode trazer implicacoes na classificacao, dependendo do tipo de classificador os dados tem de ser normalizados 
###  pois num atributo como o pH que varia entre 0 e 14, e o atributo Ammonium varia entre 0 e 1000, o que para um classificador que dependa de distancias
###  pode pesar mais uma distancia do atributo ammonium do que do pH estando assim a classificacao comprometida. ###


pH_Ammonium = data.loc[:,['Ammonium','pH']]
# ph = ph[~np.isnan(ph)]
# Ammonium = Ammonium[~np.isnan(Ammonium)]

pHnorm = (ph - ph.min())/(ph.max() - ph.min())
AmmoniumNorm = (Ammonium - Ammonium.min())/(Ammonium.max() - Ammonium.min()) 

print pHnorm
print AmmoniumNorm
# pH_Ammonium.hist()
# print np.linalg.norm(data.loc[:,'pH'])
plt.scatter(pHnorm,AmmoniumNorm)
plt.show()


