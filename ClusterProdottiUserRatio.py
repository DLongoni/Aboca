#!/usr/bin/env python

# 14-9-18: salvo questo file perchè è quello che ha generato i cluster della prima mail girata, nel caso in cui servisse replicare

# {{{ Import
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import Lib.Renders as rd
import pandas as pd
import Utils
from DA import Prodotti
from Lib import Features
from Lib import Clust
# }}}

# Get dataset
prod, prod_proc = Prodotti.get_df()

# Feature Relevance - droppo in base alle osservazioni risultanti del ciclo successivo
prod_proc.drop(['nFarma','nRight','nUsers','GeoRatio','LatVar','nReg','nTot'],axis=1,inplace=True)

prod_feat = prod_proc.drop(['ProductId','Name'],axis=1)
Features.print_relevance(prod_feat)

# Outliers
if 0: Features.print_outliers(prod_proc, prod, prod_feat.keys())

# {{{ PCA
# 1: prodotto molto consigliato
# 2: prodotto consigliato a nord
# 3: esempio consigliato correttamente e numeroso
esempi = ['P0011AN','P0018AN','P0080AB'] 
esempi_orig = prod[prod.ProductId.isin(esempi)]
esempi_proc = prod_proc[prod_proc.ProductId.isin(esempi)].drop(['ProductId','Name'],axis=1)
print('Esempi selezionati')
display(esempi_orig)
pca = PCA(n_components=2)
pca.fit(prod_feat)

pca_results = rd.pca_results(prod_feat, pca)
plt.savefig('./ClusterFig/pcadim.png')

prod_red = np.round(pca.transform(prod_feat),4)
df_red =pd.DataFrame(prod_red, columns = pca_results.index.values)

esempi_pca = pca.transform(esempi_proc)
df_samples_red =pd.DataFrame(esempi_pca, columns = pca_results.index.values)

print('Esempi trasformati')
display(pd.DataFrame(np.round(esempi_pca, 4), columns = pca_results.index.values))
# }}}

prod_visual = prod[['Ratio','NordSud','nProv','UserRatio']]
samples_visual = esempi_orig[['Ratio','NordSud','nProv','UserRatio']]

# {{{ Clustering - dati originali + Kmeans
if 1:
    for n_clusters in range(4,5):
        clusterer = KMeans(n_clusters=n_clusters)
        Clust.visualize(clusterer, n_clusters, prod_feat, esempi_proc, prod_visual, samples_visual, 1, 0)
        input('press enter')
# }}}

# {{{ Clustering - PCA + Kmeans
if 0:
    for n_clusters in range(2,5):
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        Clust.visualize(clusterer, n_clusters, df_red, df_samples_red, prod_visual, samples_visual, 1, 0)
        input('press enter')
# }}}
