#!/usr/bin/env python

# {{{ Import
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt  # NOQA
from matplotlib import ticker as ticker
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

import Utils.Renders as rd  # NOQA
from DA import Prodotti
from Utils import Clust # NOQA
from Utils import Constants
from Utils.ClAnalyzer import ClAnalyzer

from IPython import embed
# }}}

sns.set()
sns.set_palette(Constants.abc_l)
# plt.ion()

# {{{ Preparazione Dataset
df = Prodotti.get_df_group_prod(include_rare=True)
df_scaled = Prodotti.get_df_group_prod_proc(include_rare=True)
CA = ClAnalyzer(df)
CA.add_df(df_scaled, 'scaled')
feats = ['nAvSess', 'Recency', 'nUsers', 'Ratio', 'UserRatio']
feats3 = ['Recency', 'nUsers', 'Ratio']
CA.features = feats
# 1: molto consigliato
# 2: consigliato a nord
# 3: consigliato correttamente e numeroso
# samples = ['P0011AN', 'P0018AN', 'P0080AB']
samples = ['P0011AN']
CA.set_samples(samples, 'ProductId')

CA.print_relevance(df_name='scaled')
if 0:
    CA.print_outliers()
# }}}

# {{{ PCA
# pca = PCA(n_components=2).fit(CA.get_df(df_name='scaled', feat_cols=True))
cls = ['Ratio', 'Recency', 'UserRatio', 'nAvSess', 'nUsers']
df_pre_pca = CA.get_df(df_name='scaled', cols=tuple(cls))
pca = PCA(n_components=3).fit(df_pre_pca)
prod_red = np.round(pca.transform(df_pre_pca), 4)
df_red = pd.DataFrame(prod_red, columns=['Dim {}'.format(i)
                      for i in range(1, len(pca.components_)+1)])
CA.add_df(df_red, 'pca')
# pca_results = rd.pca_results(df_pre_pca, pca)
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
                                       cols=tuple(feats3)), 'ward')
        clusters3 = fcluster(linkage_m3, n_clusters, criterion='maxclust') - 1
        CA.add_cluster(clusters3, 'agglo3', n_clusters, dataset='scaled')

        Kclusterer = KMeans(n_clusters=n_clusters, random_state=1)
        Kpreds = Kclusterer.fit_predict(CA.get_df(
            df_name='scaled', feat_cols=True))
        CA.add_cluster(Kpreds, 'kmeans', n_clusters, dataset='scaled')

        Kclusterer3 = KMeans(n_clusters=n_clusters, random_state=1)
        Kpreds3 = Kclusterer3.fit_predict(CA.get_df(
            df_name='scaled', cols=tuple(feats3)))
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
# CA.print_sample_clusters('kmeans', 3)
# CA.print_cluster_diff(2, 'kmeans', 'agglo')
# CA.plot_cluster_diff(2,'agglo','kmeans',feat_cols=True)

# Grafico ad hoc presentazione
if 0:
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    CA.add_cluster_plot('kmeans3', 4, [0, 1], ax1, cols=('nUsers', 'Ratio'))
    CA.add_cluster_plot('kmeans3', 4, [0, 1], ax2, cols=('Recency', 'Ratio'))
    f.suptitle('Un esempio di suddivisione in quattro cluster', size=25)
    ax1.set_xlabel('Numero di utenti', size=20)
    ax1.set_ylabel('Correttezza', size=20)
    ax2.set_xlabel('Giorni passati dall\'ultimo consiglio (al 30/09)', size=18)
    ax2.set_ylabel('Correttezza', size=18)

    vals = [0, 0.25, 0.5, 0.75, 1]
    ax1.set_yticks(vals)
    ax2.set_yticks(vals)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax2.set_yticklabels([])
    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    flo = df[df.ProductId == 'P0011AN']
    ax1.text(flo.nUsers, flo.Ratio, 'Flora\nIntestinale', size=16,
             ha='right', va='center')
    ax2.text(flo.Recency, flo.Ratio, 'Flora\nIntestinale', size=16,
             ha='left', va='center')

    # plt.show()

if 1:  # mostro cluster crescenti
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    CA.add_cluster_plot('kmeans3', 2, [0, 1], ax1, cols=('nUsers', 'Ratio'))
    CA.add_cluster_plot('kmeans3', 3, [0, 1], ax2, cols=('nUsers', 'Ratio'))
    CA.add_cluster_plot('kmeans3', 4, [0, 1], ax3, cols=('nUsers', 'Ratio'))
    f.suptitle('Suddivisione dei prodotti in un numero crescente di '
               'cluster', size=25)
    ax1.set_title('2', size=20)
    ax2.set_title('3', size=20)
    ax3.set_title('4', size=20)
    ax1.set_ylabel('Correttezza', size=20)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    vals = [0, 0.25, 0.5, 0.75, 1]
    ax1.set_yticks(vals)
    ax2.set_yticks(vals)
    ax3.set_yticks(vals)
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax2.set_ylabel("")
    ax3.set_ylabel("")
    ax1.set_xlabel('')
    ax2.set_xlabel('Numero di utenti che hanno consigliato il '
                   'prodotto', size=20)
    ax3.set_xlabel('')
    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    ax3.tick_params(labelsize=16)
    flo = df[df.ProductId == 'P0011AN']
    ax2.text(flo.nUsers, flo.Ratio, 'Flora\nIntestinale', size=16,
             ha='right', va='center')
    # plt.show()

embed()
