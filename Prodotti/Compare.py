#!/usr/bin/env python

# {{{ Import
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
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

clust_range = range(2, 6)
if 1:
    # {{{ Clustering
    for n_clusters in clust_range:
        linkage_m = linkage(CA.get_df(df_name='scaled', feat_cols=True),
                            'ward')
        clusters = fcluster(linkage_m, n_clusters, criterion='maxclust') - 1
        CA.add_cluster(clusters, 'agglo', n_clusters, dataset='scaled')
        # print('distances for the last 5 merges:\n{}'.format(
        #     linkage_m[-5:, 2]))
        # max_d = np.mean(linkage_m[-n_clusters:len(clusters)-
        #                           (n_clusters-2), 2])
        # Clust.dendro(linkage_m, clusters, None, labels=prod.Name.values,
        #              orientation='right', max_d=max_d)
        # Clust.dendro(linkage_m, clusters, None, labels=prod.Name.values,
        #              orientation='right', max_d=max_d)

        linkage_m3 = linkage(CA.get_df(df_name='scaled',
                                       cols=tuple(features3)), 'ward')
        clusters3 = fcluster(linkage_m3, n_clusters, criterion='maxclust') - 1
        CA.add_cluster(clusters3, 'agglo3', n_clusters, dataset='scaled')

        Kclusterer = KMeans(n_clusters=n_clusters, random_state=1)
        Kpreds = Kclusterer.fit_predict(CA.get_df(
            df_name='scaled', feat_cols=True))
        CA.add_cluster(Kpreds, 'kmeans', n_clusters, dataset='scaled')

        Kclusterer3 = KMeans(n_clusters=n_clusters, random_state=1)
        Kpreds3 = Kclusterer3.fit_predict(CA.get_df(
            df_name='scaled', cols=tuple(features3)))
        CA.add_cluster(Kpreds3, 'kmeans3', n_clusters, dataset='scaled')

        pca_clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        pca_preds = pca_clusterer.fit_predict(CA.get_df(df_name='pca'))
        CA.add_cluster(pca_preds, 'kmeans_pca', n_clusters, dataset='pca')
    # }}}

# CA.plot_cluster('kmeans',2,feat_cols=True)
# CA.plot_cluster('agglo',3,df_name='scaled',feat_cols=True)
# CA.plot_cluster('kmeans_pca',3,df_name='pca')
# CA.plot_cluster('kmeans_pca',3,feat_cols=True)
CA.describe_clusters('kmeans', 2)
CA.print_sample_clusters('kmeans', 3)
CA.print_cluster_diff(2, 'kmeans', 'agglo')
# CA.plot_cluster_diff(2,'agglo','kmeans',feat_cols=True)
embed()