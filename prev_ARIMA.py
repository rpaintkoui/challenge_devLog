#%% #librairies à importer
import pandas as pd
import matplotlib.pyplot as plt
from download import download
from pandas import datetime
import numpy as np


# %% #importation jeu de donnees
url="https://docs.google.com/spreadsheets/d/e/2PACX-1vQVtdpXMHB4g9h75a0jw8CsrqSuQmP5eMIB2adpKR5hkRggwMwzFy5kB-AIThodhVHNLxlZYm8fuoWj/pub?gid=2105854808&single=true&output=csv"
path_target = "./data_prev.csv" 
download(url, path_target, replace=True)

#%%
data_raw = pd.read_csv("data_prev.csv")
data_raw = data_raw.drop(columns = [data_raw.columns[4], data_raw.columns[5]])
data_raw = data_raw.drop(index = [data_raw.index[0], data_raw.index[1]])
data_raw.columns = ["Date", "Heure", "Cumule", "Jour"]

# %% 
# 
# #traitement donnees
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
################ period 1 = 00-09 AM:
data_P1 = pd.DataFrame()
for k in range(2, len(data_raw)):
    try :
        heure = data_raw["Heure"][k].hour
        if heure < 10 and heure >= 0:
            data_P1 = data_P1.append(data_raw.loc[k])
    except : 
        continue
## on ne garde que la dernière relève quotidienne de la période
data_P1 = data_P1.drop_duplicates(subset = ['Date'], keep = 'last')
data_P1 = data_P1.set_index("Date")
cols = ["Heure", "Jour", "Cumule"]
data_P1 = data_P1.reindex(columns = cols)

#%%

################ period 2 = 09AM - 06PM:
data_P2 = pd.DataFrame()
for k in range(2, len(data_raw)):
    try :
        heure = data_raw["Heure"][k].hour
        if heure >= 9 and heure < 18:
            data_P2 = data_P2.append(data_raw.loc[k])
    except : 
        continue

## on ne garde que la dernière relève quotidienne de la période
data_P2 = data_P2.drop_duplicates(subset = ['Date'], keep = 'last')
data_P2 = data_P2.set_index("Date")
cols = ["Heure", "Jour", "Cumule"]
data_P2 = data_P2.reindex(columns = cols)
#data_P2.set_index("Date")
#%%

################ period 3 = 06PM - 00AM:
data_P3 = pd.DataFrame()
for k in range(2, len(data_raw)):
    try :
        heure = data_raw["Heure"][k].hour
        if heure >= 18 and heure <= 23:
            data_P3 = data_P3.append(data_raw.loc[k])
    except : 
        continue

## on ne garde que la dernière relève quotidienne de la période
data_P3 = data_P3.drop_duplicates(subset = ['Date'], keep = 'last')
data_P3 = data_P3.set_index("Date")
cols = ["Heure", "Jour", "Cumule"]
data_P3 = data_P3.reindex(columns = cols)
#data_P3.set_index("Date")

#%%

################################# JOUR

data_unique = data_raw.drop_duplicates(subset = ['Date'], keep = 'last')
data_unique = data_unique.set_index("Date")
cols = ["Heure", "Jour", "Cumule"]
data_unique = data_unique.reindex(columns = cols)

def cleanData(dataFrame):

    dataFrame = pd.read_csv("data_prev.csv")
    dataFrame = dataFrame.drop(columns = [dataFrame.columns[4], dataFrame.columns[5]])
    dataFrame = dataFrame.drop(index = [dataFrame.index[0], dataFrame.index[1]])
    dataFrame.columns = ["Date", "Heure", "Cumule", "Jour"]
    data_raw = data_raw.dropna(axis = 0)
    #conversion aux formats heure et date
    data_raw["Heure"] = data_raw["Heure"].apply(parse_time)
    data_raw["Date"] = data_raw["Date"].apply(parse_date)
    output = dataFrame.drop_duplicates(subset = ['Date'], keep = 'last')
    output = data_unique.set_index("Date")
    cols = ["Heure", "Jour", "Cumule"]
    output = data_unique.reindex(columns = cols)
    return(output)

# %% exportation donnees

data_P1.to_csv ('donnees_velos_P1.csv', index = True, header=True)
data_unique.to_csv ('donnees_velos_jour.csv', index = True, header=True)


# %% étudier la stationarité
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
## AVEC LES FONCTIONS D'AUTOCORRELATION ET AUTOCORRELATION PARTIELLE

### Tracé de l'ACF de la série de la période 1
data_P1_Jour = data_P1.drop(columns = ["Heure", "Cumule"])
plot_acf(data_P1_Jour)
plot_pacf(data_P1_Jour)

# %%

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

X = data_P1_Jour.values
n = int(0.8*len(X))
train = X[:n] # data considered as train data
test = X[n:]  # data considered as test data
predictions = []

#p,d,q  
# p = periods taken for autoregressive model
#d -> Integrated order, difference
# q periods in moving average model
model_arima = ARIMA(train,order=(3, 1, 6))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)

predictions_ARIMA = model_arima_fit.forecast(steps=len(test))
#print(predictions_ARIMA)
#print(test)

plt.plot(test)
plt.plot(predictions_ARIMA,color='red')

mean_squared_error(test,predictions_ARIMA)

# %%
#### recherche du tierce gagnant

import itertools
q = range(1,10)
p = range(1,10)
d = range(1,2)
pdq = list(itertools.product(p,d,q))

import warnings
warnings.filterwarnings('ignore')
crit = 534677823345
for param in pdq:
    try:
        model_arima = ARIMA(train,order=param)
        model_arima_fit = model_arima.fit()
        #print(param, model_arima_fit.aic)
        if crit > model_arima_fit.aic:
            crit = model_arima_fit.aic
            tierce_gagnant = param
    except:
        continue
    
print(tierce_gagnant)

#%%

model_arima = ARIMA(train,order=tierce_gagnant)
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)

predictions_ARIMA = model_arima_fit.forecast(steps=len(test))

plt.plot(test)
plt.plot(predictions_ARIMA,color='red')

#calcul de la somme du carré des erreurs
mean_squared_error(test,predictions_ARIMA)

# %%

# %%
