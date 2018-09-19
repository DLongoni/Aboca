#!/usr/bin/env python

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D, axes3d
import pandas as pd

data = pd.read_csv('./Dataset/VrAvatarHistory.csv')
# Drop Cs dato che è un round di CsTotal
data.drop(['Cs'],axis=1,inplace=True)
data.StartDate = pd.to_datetime(data.StartDate) # checkare se  è corretto

# Pulizia
# c'è un valore spropositatamente alto per l'unica giocata dell'utente 5019
data = data.drop(data.index[data.Sales>2000]) # valore errato
# Curiosamente ci sono dei valori molto alti di solutions, tutti corrispondenti a CsTotal = 7.8 e allo stesso utente 2018, che su 14 giocate ha totalizzato CsTotal 7.8 e un punteggio di Solutions altissimo
data = data.drop(data.index[data.Solutions>1000])

df = data[['UserId','SessionId','AvatarId','CsTotal','StartDate']]

# Recency
max_date = pd.Timestamp(2018,7,22)
df_tot = pd.DataFrame(df.UserId.unique(),columns=['UserId']) 
dfr = df.groupby('UserId')['StartDate'].max().reset_index()
df_tot = pd.merge(df_tot,dfr)
df_tot['Recency'] = max_date - df_tot.StartDate
df_tot['Recency'] = df_tot['Recency'].dt.seconds/(60*60*24)

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

# Standardize
df_log = df_tot.copy()
# df_log.Satisfaction = np.log(df_log.Satisfaction)
df_log.Recency = np.log(df_log.Recency)
df_log.Frequency = np.log(df_log.Frequency)

df_logz = df_tot.copy()
df_logz.Satisfaction = scale(df_tot.Satisfaction)
df_logz.Recency = scale(df_tot.Recency.astype(float))
df_logz.Frequency = scale(df_log.Frequency)

# df_tot.plot(x='Recency',y='Frequency',style='o')
# sns.distplot(np.log(df_tot.Recency))

range_n_clusters = list(range(2,4))
df_pred = df_logz[['Recency','Satisfaction','Frequency']]
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters).fit(df_pred)
    preds = clusterer.predict(df_pred)
    centers = clusterer.cluster_centers_
    score = silhouette_score(df_pred, preds, metric='euclidean')
    print( "For n_clusters = {0}. The average silhouette_score is : {1}".format(n_clusters, score))
    pyplot.subplot(221) 
    pyplot.scatter(x=df_tot.Recency,y=df_tot.Satisfaction,c=preds) 
    pyplot.xlabel('Gg da ultima giocata')
    pyplot.ylabel('Satisfaction')
    pyplot.subplot(222) 
    pyplot.scatter(x=df_tot.Frequency,y=df_tot.Satisfaction,c=preds) 
    pyplot.xlabel('Num giocate')
    pyplot.ylabel('Satisfaction')
    pyplot.subplot(223) 
    pyplot.scatter(x=df_tot.Recency,y=df_tot.Frequency,c=preds) 
    pyplot.xlabel('Gg da ultima giocata')
    pyplot.ylabel('Num giocate')
    pyplot.show()
    input('press enter')

# Analisi con altre variabili
# dfwm = data.groupby('UserId')['WrongMotiveScore'].sum().reset_index()
# dfwm['Preds']=preds
# dfwm.groupby('Preds').mean()

# Plot 3d
fig = pyplot.figure()
ax = Axes3D(fig)
# pltTre = pyplot.figure().gca(projection='3d')
ax.scatter(df_tot.Recency,df_tot.Frequency,df_tot.Satisfaction, c=preds)
ax.set_xlabel('Gg da ultima giocata')
ax.set_ylabel('NumGiocate')
ax.set_zlabel('Satisfaction')
pyplot.show()
