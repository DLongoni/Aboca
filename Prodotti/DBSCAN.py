#!/usr/bin/env python

# {{{ Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
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
prod_proc = Prodotti.get_df_group_prod_proc()
CA = ClAnalyzer(prod)
CA.add_df(prod_proc, 'scaled')
feats = ['nAvSess', 'Recency', 'nUsers', 'Ratio', 'UserRatio']
feats3 = ['Recency', 'nUsers', 'Ratio']
CA.features = feats
CA.print_relevance(df_name='scaled')

# 1: molto consigliato
# 2: consigliato a nord
# 3: consigliato correttamente e numeroso
samples = ['P0011AN', 'P0018AN', 'P0080AB']
CA.set_samples(samples, 'ProductId')
if 0:
    CA.print_outliers()

# PCA
pca = PCA(n_components=2).fit(CA.get_df(df_name='scaled', cols=tuple(feats3)))
prod_red = np.round(pca.transform(CA.get_df(df_name='scaled',
                                            cols=tuple(feats3))), 4)
df_red = pd.DataFrame(
    prod_red, columns=['Dim {}'.format(i)
                       for i in range(1, len(pca.components_)+1)])
CA.add_df(df_red, 'pca')

# pca_results = rd.pca_results(CA.get_df(df_name='scaled',
#                                        feat_cols=True), pca)
# plt.savefig('./Fig/pcadim.png')
# pyplot.show()
# }}}

if 1:
    clust_range = range(2, 6)
    # {{{ Clustering
    for n_clusters in clust_range:
        Kclusterer3 = KMeans(n_clusters=n_clusters, random_state=1)
        Kpreds3 = Kclusterer3.fit_predict(CA.get_df(
            df_name='scaled', cols=tuple(feats3)))
        CA.add_cluster(Kpreds3, 'kmeans3', n_clusters, dataset='scaled')

        pca_clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        pca_preds = pca_clusterer.fit_predict(CA.get_df(df_name='pca'))
        CA.add_cluster(pca_preds, 'kmeans_pca', n_clusters, dataset='pca')
    # }}}


def kdist(vals, k=4, sort=True):
    ret = np.zeros([len(vals), 1])
    dist_mat = squareform(pdist(vals))
    for i in range(0, len(vals)):
        i_distarr = dist_mat[i, :]
        i_distarr = np.sort(i_distarr)
        ret[i] = i_distarr[k]
    if sort:
        ret = np.sort(np.squeeze(ret))[::-1]
    return ret


def dbscan_labels(eps, min_samples, df):
    db_clust = DBSCAN(eps=eps, min_samples=min_samples)
    db_clust.fit(df)
    labels = db_clust.labels_
    return labels


def get_nclust(labels):
    nclust = len(set(labels)) - (1 if -1 in labels else 0)
    return nclust


k = kdist(CA.get_df(df_name='scaled', cols=tuple(feats3)), 6)
plt.plot(k)
# plt.show()
labels = dbscan_labels(1.2, 5, CA.get_df(df_name='scaled', cols=tuple(feats3)))
nclust = get_nclust(labels)
CA.add_cluster(labels, 'DBSCAN', nclust, dataset='scaled')

k_pca = kdist(CA.get_df(df_name='pca'), 6)
labels_pca = dbscan_labels(1, 10, CA.get_df(df_name='pca'))
nclust_pca = get_nclust(labels)
CA.add_cluster(labels, 'DBSCAN_pca', nclust, dataset='pca')

d = CA.get_df(df_name='scaled', cols=tuple(feats3))
NearN = NearestNeighbors(radius=1.2, metric='euclidean')
NearN.fit(d)
nb = NearN.radius_neighbors(d, 1.2)

# CA.plot_cluster('DBSCAN', 2, cols=tuple(feats3))

# CA.plot_cluster('kmeans',2,feat_cols=True)
# CA.plot_cluster('agglo',3,df_name='scaled',feat_cols=True)
# CA.plot_cluster('kmeans_pca',3,df_name='pca')
# CA.plot_cluster('kmeans_pca',3,feat_cols=True)
# CA.describe_clusters('kmeans', 2)
# CA.print_sample_clusters('kmeans', 3)
# CA.print_cluster_diff(2, 'kmeans', 'agglo')
# CA.plot_cluster_diff(2,'agglo','kmeans',feat_cols=True)
embed()
