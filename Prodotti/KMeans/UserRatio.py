#!/usr/bin/env python

# 14-9-18: salvo questo file perchè è quello che ha generato i cluster della prima mail girata, nel caso in cui servisse replicare

# {{{ Import
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import Utils.Renders as rd
import pandas as pd
from DA import Prodotti
from Utils import Features
from Utils import Clust
# }}}

# Get dataset
prod = Prodotti.get_df_group_prod()
prod_proc = Prodotti.get_df_group_prod_proc(prod)

# Feature Relevance - droppo in base alle osservazioni risultanti del ciclo successivo
prod_proc.drop(['nFarma','nRight','nUsers','GeoRatio','LatVar','nReg','nTot'],axis=1,inplace=True)
prod_feat = prod_proc.drop(['ProductId','Name'],axis=1)
Features.print_relevance(prod_feat)

# Outliers
if 0: Features.print_outliers(prod_proc, prod, prod_feat.keys())

# {{{ PCA
pca = PCA(n_components=2).fit(prod_feat)
pca_results = rd.pca_results(prod_feat, pca)
# plt.savefig('./ClusterFig/pcadim.png')
prod_red = np.round(pca.transform(prod_feat),4)
df_red =pd.DataFrame(prod_red, columns = pca_results.index.values)
# }}}

# {{{ Prodotti campione
# 1: prodotto molto consigliato
# 2: prodotto consigliato a nord
# 3: esempio consigliato correttamente e numeroso
samples = ['P0011AN','P0018AN','P0080AB'] 
samples_orig = prod[prod.ProductId.isin(samples)]
samples_proc = prod_proc[prod_proc.ProductId.isin(samples)].drop(['ProductId','Name'],axis=1)
print('samples selezionati')
display(samples_orig)

samples_pca = pca.transform(samples_proc)
df_samples_red =pd.DataFrame(samples_pca, columns = pca_results.index.values)
print('samples trasformati')
display(pd.DataFrame(np.round(samples_pca, 4), columns = pca_results.index.values))
# }}}

# Questa selezione colonne serve alla procedura di visualizzazione
prod_visual = prod[['Ratio','NordSud','nProv','UserRatio']]
samples_visual = samples_orig[['Ratio','NordSud','nProv','UserRatio']]
clust_range = range(3,4)

# {{{ Clustering - dati originali + Kmeans
if 1:
    for n_clusters in clust_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        clusterer.fit(prod_feat)
        preds = clusterer.predict(prod_feat)
        samples_preds = clusterer.predict(samples_proc)
        centers = clusterer.cluster_centers_

        Clust.visualize('KMeans', prod_feat, samples_proc, preds, samples_preds, centers, prod_visual, samples_visual, 1)
        plt.show()
        input('press enter')
# }}}

# {{{ Clustering - PCA + Kmeans
if 0:
    for n_clusters in clust_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        clusterer.fit(df_red)
        preds = clusterer.predict(df_red)
        samples_preds = clusterer.predict(df_samples_red)
        centers = clusterer.cluster_centers_

        Clust.visualize('KMeans', df_red, df_samples_red, preds, samples_preds, centers, prod_visual, samples_visual, 0)
        input('press enter')
# }}}
