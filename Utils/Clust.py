#!/usr/bin/env python

# 14-9-18: salvo questo file perchè è quello che ha generato i cluster della prima mail girata, nel caso in cui servisse replicare

# {{{ Import
import numpy as np
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import Utils.Renders as rd
import pandas as pd
from DA import Prodotti
from Utils import Features
from Utils import Constants
from IPython import embed
# }}}

# {{{ Clustering - PCA + Kmeans
def add_cluster_plot(df,df_sample,col1,col2,pred_tot,samples_pred,sub_num=None):
    col_arr = Constants.colors
    c_tot = [col_arr[i] for i in pred_tot]
    c_samp = [col_arr[i] for i in samples_pred]
    if sub_num:
        plt.subplot(sub_num) 
    plt.scatter(x=df[col1],y=df[col2],c=c_tot) 
    plt.scatter(x=df_sample[col1],y=df_sample[col2], lw=1, 
        facecolor=c_samp,marker='D',edgecolors='black') 
    plt.xlabel(col1)
    plt.ylabel(col2)

def clustplot(df, preds, df_samples, samples_preds):
    cols = df.columns
    ncolumns = len(cols)
    if ncolumns == 2:
        add_cluster_plot(df,df_samples,cols[0],cols[1],preds,samples_preds,None)
    else:
        iter_sub = [221,222,223,224]
        if ncolumns ==3:
            iter_cols = [[0,1],[2,1],[0,2],[2,0]]
        elif ncolumns ==4:
            iter_cols = [[0,1],[3,1],[0,2],[3,2]]
        elif ncolumns ==5:
            iter_cols = [[0,1],[2,3],[0,4],[2,4]]
        elif ncolumns ==6:
            iter_cols = [[0,1],[2,3],[4,5],[2,0]]
        for i in range(0,4):
            c = iter_cols[i]
            sp = iter_sub[i]
            add_cluster_plot(df,df_samples,cols[c[0]],cols[c[1]],preds,samples_preds,sp)

def visualize(clname, df_fit, samples_fit, preds, samples_preds, centers=None, df_plot=None, samples_plot=None, plot=True, save=False):
    if 0:
        clusterer.fit(df_fit)
        preds = clusterer.predict(df_fit)
        samples_preds = clusterer.predict(samples_fit)
        centers = clusterer.cluster_centers_

    n = max(preds)-min(preds)+1

    sscore = silhouette_score(df_fit, preds, random_state=1)
    csscore = calinski_harabaz_score(df_fit, preds)  
    print("Clusters: {0}, silhouette = [{1:.4f}], calinski = [{2:.4f}]".format(n, sscore, csscore))
    
    if plot:
        f = plt.figure(); axf = f.gca();
        name = 'FeatspaceSpace_' + clname + '_' + str(n)
        f.suptitle(name)
        clustplot(df_fit, preds, samples_fit, samples_preds)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if save:
            f.savefig('./Fig/'+name+'.png')

        if not df_plot is None:
            f = plt.figure(); axf = f.gca();
            name = 'OrigSpace_' + clname + '_' + str(n)
            f.suptitle(name)
            clustplot(df_plot, preds, samples_plot, samples_preds)
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            if save: 
                f.savefig('./Fig/'+name+'.png')

    return preds

def dendro_clusterer(model, **kwargs):
    children = model.children_
    distance = np.arange(children.shape[0])
    no_of_observations = np.arange(2, children.shape[0]+2)
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    clusters = model.labels_ 
    dendro(linkage_matrix,**kwargs)

def dendro(linkage_matrix, clusters, save_name=None, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d

    colors = {}
    dflt_col = "#808080"   
    for i, i12 in enumerate(linkage_matrix[:,:2].astype(int)):
        c1, c2 = (colors[x] if x > len(linkage_matrix) else Constants.colors[clusters[x]]
            for x in i12)
        colors[i+1+len(linkage_matrix)] = c1 if c1 == c2 else dflt_col

    f = plt.figure(); axf = f.gca();
    ddata = dendrogram(linkage_matrix, **kwargs, ax = axf, link_color_func = lambda k: colors[k])
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
        f.savefig('./Fig/'+save_name + '_' + nclust + '_dendro.png',dpi=900)

def describe(df, clusters):
    c_range = range(0,max(clusters)+1)
    for ic in c_range:
        print("*** Custer [{0}]".format(str(ic)))
        i_df = df.iloc[clusters==ic]
        print(i_df.describe())
# }}}
