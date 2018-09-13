#!/usr/bin/env python

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D, axes3d
import pandas as pd

data = pd.read_csv('./Dataset/Dumps/Puliti/out_VrAvatarHistory.csv')
# Drop Cs dato che è un round di CsTotal
data.drop(['Cs'],axis=1,inplace=True)
data.StartDate = pd.to_datetime(data.StartDate)

# Pulizia
# c'è un valore spropositatamente alto per l'unica giocata dell'utente 5019
data = data.drop(data.index[data.Sales>2000]) # valore errato
# Curiosamente ci sono dei valori molto alti di solutions, tutti corrispondenti a CsTotal = 7.8 e allo stesso utente 2018, che su 14 giocate ha totalizzato CsTotal 7.8 e un punteggio di Solutions altissimo
data = data.drop(data.index[data.Solutions>1000])
# Utenti prova fino al 7
data = data.drop(data.index[data.UserId<=7])

df = data[['UserId','SessionId','AvatarId','CsTotal','StartDate']]

# Recency
max_date = pd.Timestamp(2018,8,8)
df_tot = pd.DataFrame(df.UserId.unique(),columns=['UserId']) 
dfr = df.groupby('UserId')['StartDate'].max().reset_index()
df_tot = pd.merge(df_tot,dfr)
df_tot['Recency'] = max_date - df_tot.StartDate
df_tot['Recency'] = df_tot['Recency'].dt.seconds/(60*60*24) + df_tot['Recency'].dt.days

# Frequency
# in termini di numero di giorni distinti in cui un farmacista ha giocato
df = df.assign(Day = df.StartDate.dt.date)
# Test: usando il numero di sessioni cambia poco
dff = df.groupby('UserId')['Day'].nunique().reset_index()
dff.rename(columns={'Day': 'Frequency'}, inplace = True)
df_tot = pd.merge(df_tot,dff)

# Customer Statisfaction
dfc = df.groupby('UserId')['CsTotal'].mean().reset_index()
df_tot = pd.merge(df_tot,dfc)
df_tot.rename(columns={'CsTotal': 'Satisfaction'}, inplace = True)

top_players = [12,2668,4940]

