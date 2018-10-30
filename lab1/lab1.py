import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing


##########1a
data = pd.read_csv('algae.csv')
season = data['season']
ph = data['pH']
Ammonium = data['Ammonium']

##########1b
# print data.shape
# print data.iloc[0,:][3]
# print data.head()
# print data.describe()
# print data['season'].describe()
# print season.head()


###########1c
# print len([x for x in data.iloc[0,:] if type(x) == np.float64])


###########1d

#print season.describe()
#seasons = np.unique(season)
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


###########1e
# print ph.dtype
# print ph.describe()
# print [x for x in np.unique(ph) if math.isnan(x) == False]
# print len([x for x in np.unique(ph) if math.isnan(x) == True])


########### OUTLIERS
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

indices_ph_nan = np.argwhere(np.isnan(ph))
indices_Ammonium_nan = np.argwhere(np.isnan(Ammonium))

indices_nan = np.vstack((indices_ph_nan,indices_Ammonium_nan))
idx_nan = list(indices_nan.ravel())
print idx_nan

ph = ph.drop(idx_nan)
Ammonium = Ammonium.drop(idx_nan)

# ph = ph[~np.isnan(ph)]
# Ammonium = Ammonium[~np.isnan(Ammonium)]


pHnorm = (ph - ph.min())/(ph.max() - ph.min())
AmmoniumNorm = (Ammonium - Ammonium.min())/(Ammonium.max() - Ammonium.min()) 

# print "Norm1 ",pHnorm
# print "Norm1 ",AmmoniumNorm

pHnorm2 = ph / np.linalg.norm(ph)
AmmoniumNorm2 = Ammonium / np.linalg.norm(Ammonium)

# print "Norm2 ",pHnorm2
# print "Norm2 ",AmmoniumNorm2

# print "Normalizacao ", np.all(pHnorm2 == pHnorm) 

plt.figure()
plt.scatter(pHnorm2,AmmoniumNorm2)

plt.figure()
plt.scatter(pHnorm,AmmoniumNorm)
plt.show()

print data.head()
print np.cov(data.loc[:,['pH','Oxygen','Chloride','Nitrates','Ammonium','Orthophosphate','Phosphate','Chlorophyll']])


