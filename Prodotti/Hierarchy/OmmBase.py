#!/usr/bin/env python

# questo file replica il clustering che ha generato la prima mail sul clustering
# ma lo impesto cambiando le variabili e sperimentando. Le tecniche però sono
# le stesse

# {{{ Import
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import Utils.Renders as rd
import pandas as pd
from DA import Prodotti
from Utils import Clust
from Utils.ClAnalyzer import ClAnalyzer
from IPython import embed
# }}}

# {{{ Preparazione Dataset
# Get dataset
prod = Prodotti.get_df_group_prod()
prod_proc = Prodotti.get_df_group_prod_proc(prod)
CA = ClAnalyzer('Hierachy',prod)
CA.add_df(prod_proc,'scaled')
features = prod.columns.drop(['nFarma','nRight','nUsers','GeoRatio','LatVar','nReg','nTot','ProductId','Name'])
CA.features = features
CA.print_relevance(df_name='scaled')

# 1: molto consigliato
# 2: consigliato a nord
# 3: consigliato correttamente e numeroso
samples = ['P0011AN','P0018AN','P0080AB'] 
CA.set_samples(samples,'ProductId')
if 0: CA.print_outliers(df_name='scaled')

# PCA
pca = PCA(n_components=2).fit(CA.get_df_feat(df_name='scaled'))
prod_red = np.round(pca.transform(CA.get_df_feat(df_name='scaled')),4)
df_red = pd.DataFrame(prod_red, columns=['Dim {}'.format(i) for i in range(1,len(pca.components_)+1)])
CA.add_df(df_red,'pca')

# pca_results = rd.pca_results(CA.get_df_feat(df_name='scaled'), pca)
# plt.savefig('./Fig/pcadim.png')
# pyplot.show()
# }}}

clust_range = range(2,4)
if 1:
    # {{{ Clustering - dati originali + Kmeans
    for n_clusters in clust_range:
        linkage_m = linkage(CA.get_df_feat(df_name='scaled'), 'ward')
        clusters = fcluster(linkage_m, n_clusters, criterion='maxclust') - 1
        CA.add_cluster(clusters, 'agglo', n_clusters)
        print('distances for the last 5 merges:\n{}'.format(linkage_m[-5:,2]))
        # max_d = np.mean(linkage_m[-n_clusters:len(clusters)-(n_clusters-2),2])
        # Clust.dendro(linkage_m, clusters, None, labels=prod.Name.values, orientation='right', max_d=max_d)
        # Clust.dendro(linkage_m, clusters, None, labels=prod.Name.values, orientation='right', max_d=max_d)

        Kclusterer = KMeans(n_clusters=n_clusters, random_state=1)
        Kpreds = Kclusterer.fit_predict(CA.get_df_feat(df_name='scaled'))
        Kcenters = Kclusterer.cluster_centers_
        CA.add_cluster(Kpreds, 'kmeans', n_clusters)

        # plt.show()
        # input('press enter')
# }}}

CA.visualize('kmeans',2)
CA.visualize('agglo',3,df_name='scaled')
embed()
if 0:
# {{{ Clustering - PCA + Kmeans
    for n_clusters in clust_range:
        clusterer = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
        Clust.visualize(clusterer, n_clusters, df_red, df_samples_red, prod_visual, samples_visual, 0)
        input('press enter')
# }}}
