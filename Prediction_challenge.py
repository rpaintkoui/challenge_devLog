#%% #packages utilisés (à installer éventuellement)
!pip install pandas
!pip install download
!pip install matplotlib
!pip install numpy
!pip install statsmodels
!pip install pmdarima

#%% #librairies à importer
import pandas as pd
import matplotlib.pyplot as plt
from download import download
from pandas import datetime
import numpy as np


# %% 

################################################### 
#        PARTIE 1 - TRAITEMENT DES DONNEES        #
###################################################

# #téléchargement du jeu de donnees

url="https://docs.google.com/spreadsheets/d/e/2PACX-1vQVtdpXMHB4g9h75a0jw8CsrqSuQmP5eMIB2adpKR5hkRggwMwzFy5kB-AIThodhVHNLxlZYm8fuoWj/pub?gid=2105854808&single=true&output=csv"
path_target = "./data_prev.csv" 
download(url, path_target, replace=True)

#%%
#importation des donnees
data_raw = pd.read_csv("data_prev.csv")
data_raw = data_raw.drop(columns = [data_raw.columns[4], data_raw.columns[5]])
data_raw = data_raw.drop(index = [data_raw.index[0], data_raw.index[1]])
data_raw.columns = ["Date", "Heure", "Cumule", "Jour"]

# %% 
# 
# traitement donnees
def parse_time(x):
    return datetime.strptime(str(x), "%H:%M:%S").time()

def parse_date(x):
    try:
        return datetime.strptime(str(x), "%Y-%m-%d").date()
    except :
        return datetime.strptime(str(x), "%d/%m/%Y").date()
 
data_raw = data_raw.dropna(axis = 0)

#conversion aux formats heure et date
data_raw["Heure"] = data_raw["Heure"].apply(parse_time)
data_raw["Date"] = data_raw["Date"].apply(parse_date)

####split the data into periods

#%%
################ periode 1 = 00-09h:
data_P1 = pd.DataFrame()
for k in range(2, len(data_raw)):
    try :
        heure = data_raw["Heure"][k].hour
        if heure < 9 and heure >= 0:
            data_P1 = data_P1.append(data_raw.loc[k])
        if heure == 9 and data_raw["Heure"][k].minute <11 :
            data_P1 = data_P1.append(data_raw.loc[k])
    except : 
        continue
## on ne garde que la dernière relève quotidienne de la période
data_P1 = data_P1.drop_duplicates(subset = ['Date'], keep = 'last')
data_P1 = data_P1.set_index("Date")
data_P1.index = pd.to_datetime(data_P1.index, format = "%Y-%m-%d")
data_P1 = data_P1.resample('D').ffill()
cols = ["Heure", "Jour", "Cumule"]
data_P1 = data_P1.reindex(columns = cols)

data_P1_Jour = data_P1.drop(columns = ["Heure", "Cumule"])

#%%
################ JOUR

data_unique = data_raw.drop_duplicates(subset = ['Date'], keep = 'last')
data_unique = data_unique.set_index("Date")
cols = ["Heure", "Jour", "Cumule"]
data_unique = data_unique.reindex(columns = cols)

data_Jour = data_unique.drop(columns = ['Heure', 'Cumule'])
# %% exportation donnees

#data_P1.to_csv ('donnees_velos_P1.csv', index = True, header=True)
#data_unique.to_csv ('donnees_velos_jour.csv', index = True, header=True)


# %% 

################################################### 
#          PARTIE 2 - ETUDE DE LA SERIE           #
###################################################

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

####### SERIE DES DONNEES DE LA PERIODE 1 #######

plot_acf(data_P1_Jour, title = "Période 1 - ACF")
plot_pacf(data_P1_Jour, title = "Période 1 - PACF")

dataP1_diff = data_P1_Jour.diff(periods = 1)


plot_acf(dataP1_diff[1:], title = "Période 1_diff - ACF")

################# SERIE ENTIERES #################

plot_acf(data_Jour, title = "Entière - ACF")
plot_pacf(data_Jour, title = "Entière - PACF")

dataJour_diff = data_Jour.diff(periods = 1)

plot_acf(dataJour_diff[1:], title = "Entière_diff - ACF")

# %% 

################################################### 
#             PARTIE 3 - PREDICTION               #
###################################################

################     PERIODE 1     ################
########### RECHERCHE DES COEFFICIENTS ############

import pmdarima as pmd
from pmdarima import auto_arima

df = data_P1_Jour.copy()
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df = df.resample('D').ffill()


#on va tester le modèle sur 20% des données qu'on a
size = int(len(df) * 0.8) 

train=df.iloc[:size]
test=df.iloc[size:]

#%%
#calcul des coefficients (d,p,q) du modèle ARIMA
coeffsP1 = pmd.auto_arima(df, trace=True, suppress_warnings=True)
coeffsP1.summary()

#%%
from statsmodels.tsa.arima.model import ARIMA

#le meilleur modèle est (d,p,q) = (0,1,3) d'après la fonction auto_arima
model = ARIMA(df, order=(0,1,3))
result = model.fit()
result.summary()

start = len(train)
end = len(train) + len(test) - 1
  
#On vérifie qu'on retrouve '"bien" les données observées
predictions = result.predict(start,end,typ = 'levels').rename("Predictions")
  
# Tracé des courbes de prédictions et d'observations
test.plot(legend = True)
predictions.plot(legend = True,color = 'red')

# %%

###################     JOUR     ##################
########### RECHERCHE DES COEFFICIENTS ############

dfJour = data_Jour.copy()

#on va tester le modèle sur 20% des données qu'on a
sizeJour = int(len(dfJour) * 0.8) 

trainJour=dfJour.iloc[:sizeJour]
testJour=dfJour.iloc[sizeJour:]

#calcul des coefficients (p,d,q) du modèle ARIMA
coeffsJour = pmd.auto_arima(dfJour, trace=True, suppress_warnings=True)
coeffsJour.summary()

#%%

#le meilleur modèle est (p,d,q) = (2,1,3) d'après la fonction auto_arima
modelJour = ARIMA(dfJour, order=(2,1,3))
resultJour = modelJour.fit()
resultJour.summary()

startJour = len(trainJour)
endJour = len(trainJour) + len(testJour) - 1
  
#On vérifie qu'on retrouve '"bien" les données observées
predictionsJour = resultJour.predict(endJour,endJour+2,typ = 'levels').rename("Predictions_Jour")
  
# Tracé des courbes de prédictions et d'observations
testJour.plot(legend = True)
predictionsJour.plot(legend = True,color = 'red')
# %%


#########################################
#########################################
##                                     ##
##     PREDICTIONS POUR LE 2 AVRIL     ##
##                                     ##
#########################################
#########################################


predictions_2Avril = result.predict(end,end+10,typ = 'levels').rename("Predictions")
print("Le 2 avril, entre minuit et 09 heures, il passera {0} vélos par le compteur Albert 1er. ".format(int(predictions_2Avril.loc['2021-04-02'])))
# %%

# %%

# %%
