#!/usr/bin/env python

# {{{ Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import Utils.Renders as rd  # NOQA
import pandas as pd
from DA import Prodotti
from Utils import Clust # NOQA
from Utils.ClAnalyzer import ClAnalyzer
from IPython import embed
# }}}

# {{{ Preparazione Dataset
prod = Prodotti.get_df_group_prod()
prod_proc = Prodotti.get_df_group_prod_proc(prod)
CA = ClAnalyzer(prod)
CA.add_df(prod_proc, 'scaled')
features = prod.columns.drop(['nFarma', 'nRight', 'nUsers', 'GeoRatio',
                              'LatVar', 'nReg', 'nTot', 'ProductId', 'Name'])
features3 = features.drop(['NordSud'])
CA.features = features
CA.print_relevance(df_name='scaled')

# 1: molto consigliato
# 2: consigliato a nord
# 3: consigliato correttamente e numeroso
samples = ['P0011AN', 'P0018AN', 'P0080AB']
CA.set_samples(samples, 'ProductId')
if 0:
    CA.print_outliers()

# PCA
pca = PCA(n_components=2).fit(CA.get_df(df_name='scaled', feat_cols=True))
prod_red = np.round(pca.transform(CA.get_df(df_name='scaled',
                                            feat_cols=True)), 4)
df_red = pd.DataFrame(
    prod_red, columns=['Dim {}'.format(i)
                       for i in range(1, len(pca.components_)+1)])
CA.add_df(df_red, 'pca')

# pca_results = rd.pca_results(CA.get_df(df_name='scaled',
#                                        feat_cols=True), pca)
# plt.savefig('./Fig/pcadim.png')
# pyplot.show()
# }}}

if 0:
    clust_range = range(2, 6)
    # {{{ Clustering
    for n_clusters in clust_range:
        Kclusterer3 = KMeans(n_clusters=n_clusters, random_state=1)
        Kpreds3 = Kclusterer3.fit_predict(CA.get_df(
            df_name='scaled', cols=tuple(features3)))
        CA.add_cluster(Kpreds3, 'kmeans3', n_clusters, dataset='scaled')

        pca_clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        pca_preds = pca_clusterer.fit_predict(CA.get_df(df_name='pca'))
        CA.add_cluster(pca_preds, 'kmeans_pca', n_clusters, dataset='pca')
    # }}}


def kdist(vals, k=4):
    ret = np.zeros([len(vals), 1])
    dist_mat = squareform(pdist(vals))
    for i in range(0, len(vals)):
        i_distarr = dist_mat[i, :]
        i_distarr = np.sort(i_distarr)
        ret[i] = i_distarr[k]
    return ret


k = kdist(CA.get_df(df_name='scaled', cols=tuple(features3)))
ks = np.sort(np.squeeze(k))[::-1]
plt.plot(ks)
# plt.show()
db_clust = DBSCAN(eps=0.7, min_samples=4)
db_clust.fit(CA.get_df(df_name='scaled', cols=tuple(features3)))
labels = db_clust.labels_
nclust = len(set(labels)) - (1 if -1 in labels else 0)
CA.add_cluster(labels, 'DBSCAN', nclust, dataset='scaled')
CA.plot_cluster('DBSCAN', 2, cols=tuple(features3))

# CA.plot_cluster('kmeans',2,feat_cols=True)
# CA.plot_cluster('agglo',3,df_name='scaled',feat_cols=True)
# CA.plot_cluster('kmeans_pca',3,df_name='pca')
# CA.plot_cluster('kmeans_pca',3,feat_cols=True)
# CA.describe_clusters('kmeans', 2)
# CA.print_sample_clusters('kmeans', 3)
# CA.print_cluster_diff(2, 'kmeans', 'agglo')
# CA.plot_cluster_diff(2,'agglo','kmeans',feat_cols=True)
embed()
