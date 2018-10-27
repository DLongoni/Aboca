#!/usr/bin/env python

# {{{ Import
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import Lib.Renders as rd
import pandas as pd
from DA import Prodotti
from Lib import Features
from Lib import Clust
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
samples_orig = prod[prod.ProductId.isin(samples)]
samples_proc = prod_proc[prod_proc.ProductId.isin(samples)].drop(['ProductId','Name'],axis=1)
print('samples selezionati')
display(samples_orig)

samples_pca = pca.transform(samples_proc)
df_samples_red =pd.DataFrame(samples_pca, columns = pca_results.index.values)
print('samples trasformati')
display(pd.DataFrame(np.round(samples_pca, 4), columns = pca_results.index.values))
# }}}

def add_cluster_plot(df,df_sample,col1,col2,pred_tot,sample_pred,sub_num):
    col_arr = ['b','g','r','c','m','y']
    c_tot = [col_arr[i] for i in pred_tot]
    c_samp = [col_arr[i] for i in sample_pred]
    if sub_num:
        plt.subplot(sub_num) 
    plt.scatter(x=df[col1],y=df[col2],c=c_tot) 
    plt.scatter(x=df_sample[col1],y=df_sample[col2], lw=1, 
        facecolor=c_samp,marker='D',edgecolors='black') 
    plt.xlabel(col1)
    plt.ylabel(col2)

# {{{ Clustering - dati originali 
data_to_fit = prod_feat
samples_to_fit = samples_proc
if 1:
    for n_clusters in range(2,5):
        clusterer = GaussianMixture(n_components=n_clusters, init_params='random').fit(data_to_fit)
        preds = clusterer.predict(data_to_fit)
        sample_preds = clusterer.predict(samples_to_fit)
        score = silhouette_score(data_to_fit, preds, random_state=1)
        print("{0} clusters: {1:.4f}".format(n_clusters, score))
        
        if 1:
            add_cluster_plot(prod,samples_orig,'Ratio','NordSud',preds,sample_preds,221)
            add_cluster_plot(prod,samples_orig,'UserRatio','nProv',preds,sample_preds,222)
            add_cluster_plot(prod,samples_orig,'Ratio','nProv',preds,sample_preds,223)
            add_cluster_plot(prod,samples_orig,'NordSud','UserRatio',preds,sample_preds,224)
            fname = 'ProdOrig' + str(n_clusters) + '.png'
            # plt.savefig('./Fig/'+fname)
            plt.show()
            input('press enter')
# }}}

# {{{ Clustering - PCA 
data_to_fit = df_red
samples_to_fit = df_samples_red
if 0:
    for n_clusters in range(2,5):
        clusterer = GaussianMixture(n_components=n_clusters, init_params='random').fit(data_to_fit)
        preds = clusterer.predict(data_to_fit)
        sample_preds = clusterer.predict(samples_to_fit)
        score = silhouette_score(data_to_fit, preds, random_state=1)
        print("{0} clusters: {1:.4f}".format(n_clusters, score))
        
        if 1:
            plt.figure(0)
            fname = 'ProdPCAdomPCA' + str(n_clusters) + '.png'
            # add_cluster_plot(df_red,df_samples_red,'Dimension 1','Dimension 2',preds,sample_preds,None)
            rd.plot_gmm(clusterer,df_red.values,preds)
            # plt.title(str(n_clusters)+' clust PCA ')
            # plt.savefig('./Fig/'+fname)
            plt.figure(1)
            add_cluster_plot(prod,samples_orig,'Ratio','NordSud',preds,sample_preds,221)
            add_cluster_plot(prod,samples_orig,'UserRatio','nProv',preds,sample_preds,222)
            add_cluster_plot(prod,samples_orig,'Ratio','nProv',preds,sample_preds,223)
            add_cluster_plot(prod,samples_orig,'NordSud','UserRatio',preds,sample_preds,224)
            plt.title(str(n_clusters)+' clust PCA ')
            fname = 'ProdPCAdomOrig' + str(n_clusters) + '.png'
            # plt.savefig('./Fig/'+fname)
            plt.show()
            input('press enter')
# }}}
