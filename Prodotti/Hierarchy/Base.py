#!/usr/bin/env python

# questo file replica il clustering che ha generato la prima mail sul clustering
# ma lo impesto cambiando le variabili e sperimentando. Le tecniche però sono
# le stesse

# {{{ Import
import numpy as np
# from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
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
# plt.savefig('./Fig/pcadim.png')
prod_red = np.round(pca.transform(prod_feat),4)
df_red =pd.DataFrame(prod_red, columns = pca_results.index.values)
# }}}

# {{{ Prodotti campione
# 1: prodotto molto consigliato
# 2: prodotto consigliato a nord
# 3: esempio consigliato correttamente e numeroso
samples = ['P0011AN','P0018AN','P0080AB'] 
samples_id = [i for i in range(0,len(prod_proc)) if prod_proc.ProductId.iloc[i] in samples]
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
prod_visual = prod[prod_feat.columns]
samples_visual = samples_orig[prod_feat.columns]
clust_range = range(3,4)

if 1:
# {{{ Clustering - dati originali + Kmeans
    for n_clusters in clust_range:
        # clusterer = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
        # preds = clusterer.fit_predict(prod_feat)
        # samples_preds = clusterer.fit_predict(samples_proc)
        # Clust.visualize('Agglo', prod_feat, samples_proc, preds, samples_preds, None, prod_visual, samples_visual, 0)

        linkage_m = linkage(prod_feat, 'ward')
        clusters = fcluster(linkage_m, n_clusters, criterion='maxclust') - 1
        samples_preds = clusters[samples_id]
        print('distances for the last 5 merges:\n{}'.format(linkage_m[-5:,2]))
        max_d = np.mean(linkage_m[-n_clusters:-(n_clusters-2),2])
        Clust.visualize('Agglo', prod_feat, samples_proc, clusters, samples_preds, None, prod_visual, samples_visual, 1)
        # Clust.dendro(linkage_m, clusters, None, labels=prod.Name.values, orientation='right', max_d=max_d)
        # Clust.dendro(linkage_m, clusters, None, labels=prod.Name.values, orientation='right', max_d=max_d)

        Kclusterer = KMeans(n_clusters=n_clusters, random_state=1)
        Kclusterer.fit(prod_feat)
        Kpreds = Kclusterer.predict(prod_feat)
        Ksamples_preds = Kclusterer.predict(samples_proc)
        Kcenters = Kclusterer.cluster_centers_

        Clust.visualize('KMeans', prod_feat, samples_proc, Kpreds, Ksamples_preds, Kcenters, prod_visual, samples_visual, 1)

        plt.show()
        input('press enter')
# }}}

if 0:
# {{{ Clustering - PCA + Kmeans
    for n_clusters in clust_range:
        clusterer = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
        Clust.visualize(clusterer, n_clusters, df_red, df_samples_red, prod_visual, samples_visual, 0)
        input('press enter')
# }}}
