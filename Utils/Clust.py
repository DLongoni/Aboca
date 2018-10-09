#!/usr/bin/env python

# 14-9-18: salvo questo file perchè è quello che ha generato i cluster della
# prima mail girata, nel caso in cui servisse replicare

# {{{ Import
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from matplotlib import pyplot as plt
from Utils import Constants
from IPython import embed #NOQA
# }}}

sns.set()


# {{{ Clustering - PCA + Kmeans
def add_cluster_plot(df, df_sample, iter_cols, pred_tot, samples_pred,
                     centers, sub_num=None):
    col1 = df.columns[iter_cols[0]]
    col2 = df.columns[iter_cols[1]]
    n = max(pred_tot)-min(pred_tot)+1
    col_arr = Constants.colors
    c_tot = [col_arr[i] for i in pred_tot]
    c_samp = [col_arr[i] for i in samples_pred]
    if sub_num:
        plt.subplot(sub_num)
    plt.scatter(x=df[col1], y=df[col2], c=c_tot)
    plt.scatter(x=df_sample[col1], y=df_sample[col2], lw=1,
                facecolor=c_samp, marker='D', edgecolors='black')
    if centers is not None:
        plt.scatter(centers[:, iter_cols[0]], centers[:, iter_cols[1]], lw=1,
                    facecolor=Constants.colors[0:n], marker='X',
                    edgecolors='k', s=150)

    plt.xlabel(col1)
    plt.ylabel(col2)


def clustplot(df, preds, df_samples, samples_preds, centers):
    ncolumns = len(df.columns)
    if ncolumns == 2:
        add_cluster_plot(df, df_samples, 0, 1, preds,
                         samples_preds, centers, None)
    else:
        iter_sub = [221, 222, 223, 224]
        if ncolumns == 3:
            iter_cols = [[0, 1], [2, 1], [0, 2], [2, 0]]
        elif ncolumns == 4:
            iter_cols = [[0, 1], [3, 1], [0, 2], [3, 2]]
        elif ncolumns == 5:
            iter_cols = [[0, 1], [2, 3], [0, 4], [2, 4]]
        elif ncolumns == 6:
            iter_cols = [[0, 1], [2, 3], [4, 5], [2, 0]]
        for i in range(0, 4):
            sp = iter_sub[i]
            add_cluster_plot(df, df_samples, iter_cols[i], preds,
                             samples_preds, centers, sp)


def visualize(clname, df_fit, samples_fit, preds, samples_preds, centers=None,
              df_plot=None, samples_plot=None, save=False):
    n = max(preds)-min(preds)+1
    sscore = silhouette_score(df_fit, preds, random_state=1)
    csscore = calinski_harabaz_score(df_fit, preds)
    print("Clusters: {0}, silhouette = [{1:.4f}], calinski = "
          "[{2:.4f}]".format(n, sscore, csscore))

    f = plt.figure()
    name = 'FeatspaceSpace_' + clname + '_' + str(n)
    f.suptitle(name)
    clustplot(df_fit, preds, samples_fit, samples_preds, centers)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    if save:
        f.savefig('./Fig/'+name+'.png')

    if df_plot is not None:
        f = plt.figure()
        name = 'OrigSpace_' + clname + '_' + str(n)
        f.suptitle(name)
        clustplot(df_plot, preds, samples_plot, samples_preds, None)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if save:
            f.savefig('./Fig/'+name+'.png')

    return preds


def dendro_clusterer(model, **kwargs):
    children = model.children_
    distance = np.arange(children.shape[0])
    no_of_observations = np.arange(2, children.shape[0]+2)
    linkage_matrix = np.column_stack([children, distance, no_of_observations]
                                     ).astype(float)
    dendro(linkage_matrix, **kwargs)


def dendro(linkage_matrix, clusters, save_name=None, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d

    colors = {}
    dflt_col = "#808080"
    for i, i12 in enumerate(linkage_matrix[:, :2].astype(int)):
        c1, c2 = (colors[x] if x > len(linkage_matrix) else
                  Constants.colors[clusters[x]] for x in i12)
        colors[i+1+len(linkage_matrix)] = c1 if c1 == c2 else dflt_col

    f = plt.figure()
    axf = f.gca()
    ddata = dendrogram(linkage_matrix, **kwargs, ax=axf,
                       link_color_func=lambda k: colors[k])
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        x = d[1]
        y = 0.5 * sum(i[1:3])
        axf.plot(x, y, 'o', c=c)
        axf.annotate("%.3g" % x, (x, y), ha='left', va='bottom')

    if max_d:
        axf.axvline(x=max_d, c='k')

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    if save_name:
        nclust = str(max(clusters)+1)
        f.savefig('./Fig/'+save_name + '_' + nclust + '_dendro.png', dpi=900)


def describe(df, clusters):
    c_range = range(0, max(clusters)+1)
    for ic in c_range:
        print("*** Custer [{0}]".format(str(ic)))
        i_df = df.iloc[clusters == ic]
        print(i_df.describe())


def cluster_diff(df_orig, df_clust, cl1, cl2, name1, name2):
    clmap = cluster_mapping(cl1, cl2)
    print("Cluster mapping: [{0}]".format(clmap))
    s1 = silhouette_score(df_clust, cl1)
    c1 = calinski_harabaz_score(df_clust, cl1)
    s2 = silhouette_score(df_clust, cl2)
    c2 = calinski_harabaz_score(df_clust, cl2)
    print("Silhouette: [{0}]:[{1:.4f}]; [{2}]:[{3:.4f}]".format(
      name1, s1, name2, s2))
    print("Calinski:   [{0}]:[{1:.4f}]; [{2}]:[{3:.4f}]".format(
      name1, c1, name2, c2))
    for i in range(0, len(clmap)):
        print("***** Comparing cluster [{0}-{1}] with cluster [{2}-{3}]".
              format(clmap[i, 0], name1, clmap[i, 1], name2))
        print("Describing cluster [{0}-{1}]".format(clmap[i, 0], name1))
        print(df_orig.iloc[cl1 == i].describe())
        print("Describing cluster [{0}-{1}]".format(clmap[i, 1], name2))
        print(df_orig.iloc[cl2 == i].describe())
        cl1notcl2 = np.logical_and(cl1 == clmap[i, 0],
                                   np.logical_not(cl2 == clmap[i, 1]))
        if cl1notcl2.any():
            print("Points that are in [{0}] but not in [{1}]".format(
              name1, name2))
            print(df_orig.iloc[cl1notcl2])
        else:
            print("All points that are in [{0}] are in [{1}]".format(
              name1, name2))

        cl2notcl1 = np.logical_and(cl2 == clmap[i, 1],
                                   np.logical_not(cl1 == clmap[i, 0]))
        if cl2notcl1.any():
            print("Points that are in [{0}] but not in [{1}]".format(
              name2, name1))
            print(df_orig.iloc[cl2notcl1])
        else:
            print("All points that are in [{0}] are in [{1}]".format(
              name2, name1))


def cluster_mapping(cl1, cl2):
    n1 = max(cl1)-min(cl1)+1
    n2 = max(cl2)-min(cl2)+1
    if n1 != n2:
        raise Exception("Different number of clusters")
    clmap = np.zeros([n1, 2])
    for i in range(0, n1):
        iSum = 0
        for j in range(0, n1):
            nComp = np.logical_and(cl1 == i, cl2 == j)
            if sum(nComp) > iSum:
                iSelected = j
                iSum = sum(nComp)
        if iSum < 0.5 * sum(cl1 == i):
            raise Exception("Can't determine cluster mapping")
        clmap[i, 0] = int(i)
        clmap[i, 1] = int(iSelected)
    return clmap
# }}}
