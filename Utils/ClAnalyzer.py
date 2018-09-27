#!/usr/bin/env python

# {{{ Import
import numpy as np
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from Utils import Features
from Utils import Constants
from IPython import embed
from functools import lru_cache
from DA import Prodotti
# }}}

# The idea is to make it independent on the clusterer
class ClAnalyzer:
    BASE_DF = 'base'

    def __init__(self, name, df):
        self._name = name
        self._df_dic = {self.BASE_DF: df}
        self._clust_dic = {}
        self._samples_column = 'index'
        self._samples_ids = None
        self._features = None

    # {{{ Properties
    @property
    def name(self):
        return self._name

    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, value):
        self._features = value

    @property
    def samples_ids(self):
        return self._samples_ids
    @samples_ids.setter
    def samples_ids(self, value):
        self.get_df_samples.cache_clear()
        self._samples_ids = value

    @property
    def n(self):
        return len(self._df_dic[self.BASE_DF])

    # }}}

    # {{{ Setters
    def add_df(self, df, name):
        if name in self._df_dic:
            raise KeyError('Dataframe [{0}] already in dictionary'.format(name))
        if (not (df.index == self._df_dic[self.BASE_DF].index).all()):
            raise ValueError("The original and scaled df indices must be the same")
        self._df_dic[name] = df

    def set_samples(self, samples_labels, df_column=None):
        if df_column:
            if not df_column in self.get_df().columns:
                raise Exception("Column [{0}] not in df".format(df_column))
            self._samples_column = df_column
            basedf=self._df_dic[self.BASE_DF]
            self.samples_ids = basedf[basedf[df_column].isin(samples_labels)].index
        else:
            self._samples_column = 'index'
            self.samples_ids = samples_labels

    def add_cluster(self, clust_arr, clust_name, n_clust):
        key = self.__combine_name_n(clust_name, n_clust)
        if key in self._clust_dic:
            raise ValueError("Cluster with name [{0}] and n [{1}] " \
                "already in the dictionary".format(clust_name,n_clust))
        if len(clust_arr) != self.n:
            raise ValueError("Cluster array have [{0}] elements, " \
                "but the dataset have [{1}]".format(len(clust_arr),self.n))
        self._clust_dic[key] = clust_arr
    # }}}

    # {{{ Getters
    def get_df(self, **kwargs):
        df_name = kwargs.get("df_name",self.BASE_DF)
        if not df_name in self._df_dic:
            raise KeyError('Dataframe [{0}] not set'.format(df_name))
        return self._df_dic[df_name]

    @lru_cache(maxsize=10)
    def get_df_samples(self, feat=False, **kwargs):
        if self._samples_ids is None:
            raise Exception("Samples ids are not set")
        if feat:
            return self.get_df_feat(**kwargs).loc[self._samples_ids]
        else:
            return self.get_df(**kwargs).loc[self._samples_ids]

    @lru_cache(maxsize=10)
    def get_df_feat(self, **kwargs):
        if self._features is None:
            raise Exception("Set features first")
        return self.get_df(**kwargs)[self._features]

    @lru_cache(maxsize=10)
    def get_cluster(self, clust_name, n_clust):
        key = self.__combine_name_n(clust_name, n_clust)
        if not key in self._clust_dic:
            raise KeyError("Cluster with name [{0}] and n [{1}] " \
                "not in the dictionary".format(clust_name,n_clust))
        return self._clust_dic[key]

    @lru_cache(maxsize=10)
    def get_samples_cluster(self, clust_name, n_clust):
        cl_tot = self.get_cluster(clust_name, n_clust)
        return cl_tot[self._samples_ids]
    # }}}

    # {{{ Features
    def print_relevance(self, **kwargs):
        df = self.get_df_feat(**kwargs)
        if self._features is None:
            raise Exception("Set features first")
        cols = self._features
        print("Relevance analysis for features [{0}]".format(' '.join(cols)))
        for col in cols:
            new_data = df.drop(col, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(new_data, df[col], test_size=0.25, random_state=1)
            regressor = DecisionTreeRegressor(random_state=1)
            regressor.fit(X_train, y_train)
            score = regressor.score(X_test, y_test)
            print('Variable ', col,' predictability: ', score)

    def print_outliers(self, mode='iqr', **kwargs):
        df_out = self.get_df_feat(**kwargs)
        df_plot = self.get_df()
        if self._features is None:
            raise Exception("Set features first")
        cols = self._features
        print("Outlier analysis for features [{0}]".format(' '.join(cols)))
        for col in cols:
            if mode == 'iqr':
                Q1 = np.percentile(df_out[col], 25)
                Q3 = np.percentile(df_out[col], 75)
                step = (Q3 - Q1) * 1.5
                print("Data points considered step-outliers for the col '{}':".format(col))
                print(df_plot[~((df_out[col] >= Q1 - step) & (df_out[col] <= Q3 + step))])
            elif mode == '2perc':
                Qmin = np.percentile(df[col], 2)
                Qmax = np.percentile(df[col], 98)
                print("Data points considered 2-percent outliers for the col '{}':".format(col))
                print(df_plot[~((df_out[col] >= Qmin) & (df_out[col] <= Qmax))])
            else:
                raise ValueError('Mode should be either iqr or 2perc')
    # }}}

    # {{{ Print clusters
    def describe_clusters(self, clust_name, n_clust, **kwargs):
        clust      = self.get_cluster(clust_name, n_clust)
        df         = self.get_df(**kwargs)
        c_range    = range(0, n_clust)
        print("*** Describing clustering [{0}] with [{1}] clusters ***"
            .format(clust_name, n_clust))
        for ic in c_range:
            print("*** Custer labeled [{0}]".format(str(ic)))
            i_df = df.iloc[clust==ic]
            print(i_df.describe())

    def print_sample_clusters(self, clust_name, n_clust, **kwargs):
        samples_clust = self.get_samples_cluster(clust_name, n_clust)
        df            = self.get_df(**kwargs)
        s_range       = range(0, len(self._samples_ids))
        print("*** Samples ***")
        print(self.get_df_samples(**kwargs))
        for i_s in s_range:
            print("Sample [{0}] is in cluster [{1}]"
                .format(self._samples_ids[i_s], samples_clust[i_s]))

    def print_cluster_diff(self, n_clust, name1, name2, df_name=None):
        cl1     = self.get_cluster(name1, n_clust)
        cl2     = self.get_cluster(name2, n_clust)
        clmap   = self.__cluster_mapping(cl1,cl2)
        df_orig = self.get_df()
        print("*** Comparing cluster [{0}] and [{1}] for [{2}] clusters ***"
            .format(name1, name2, n_clust))
        print("Cluster mapping: [\n{0}\n]".format(clmap))
        if not df_name is None:
            df_score = self.get_df_feat(df_name=df_name)
            s1 = silhouette_score(df_score, cl1)
            c1 = calinski_harabaz_score(df_score, cl1)  
            s2 = silhouette_score(df_score, cl2)
            c2 = calinski_harabaz_score(df_score, cl2)  
            print("Silhouette: [{0}]:[{1:.4f}]; [{2}]:[{3:.4f}]".format(name1,s1,name2,s2)) 
            print("Calinski:   [{0}]:[{1:.4f}]; [{2}]:[{3:.4f}]".format(name1,c1,name2,c2)) 
        for i in range(0,len(clmap)):
            print("***** Comparing cluster [{0}-{1}] with cluster [{2}-{3}]".
                format(clmap[i,0],name1,clmap[i,1],name2))
            print("Describing cluster [{0}-{1}]".format(clmap[i,0],name1))
            print(df_orig.iloc[cl1==i].describe())
            print("Describing cluster [{0}-{1}]".format(clmap[i,1],name2))
            print(df_orig.iloc[cl2==i].describe())
            cl1notcl2 = np.logical_and(cl1==clmap[i,0], np.logical_not(cl2==clmap[i,1]))
            if cl1notcl2.any():
                print("Points that are in [{0}] but not in [{1}]".format(name1,name2))
                print(df_orig.iloc[cl1notcl2])
            else:
                print("All points that are in [{0}] are in [{1}]".format(name1,name2))

            cl2notcl1 = np.logical_and(cl2==clmap[i,1],np.logical_not(cl1==clmap[i,0]))
            if cl2notcl1.any():
                print("Points that are in [{0}] but not in [{1}]".format(name2,name1))
                print(df_orig.iloc[cl2notcl1])
            else:
                print("All points that are in [{0}] are in [{1}]".format(name2,name1))
    # }}}

    # {{{ Plot clusters
    def visualize(self, cl_name, n_clust, centers=None, save=False, **kwargs):
        f = plt.figure(); axf = f.gca();
        df_name = kwargs.get("df_name",self.BASE_DF)
        plotname = cl_name + "_" + str(n_clust) + "_" + df_name 
        f.suptitle(plotname)

        ncol = len(self._features)
        if ncol == 2:
            self.add_cluster_plot(cl_name, n_clust, [0,1], centers, None, **kwargs)
        else:
            iter_sub  = [221,222,223,224]
            iter_cols = self.__select_plot_cols(ncol)
            for i in range(0,4):
                sp = iter_sub[i]
                self.add_cluster_plot(cl_name, n_clust, iter_cols[i], centers, sp, **kwargs)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if save:
            f.savefig(Constants.pic_path+plotname+'.png')
        plt.show()

    def add_cluster_plot(self, cl_name, n_clust, iter_cols, centers, sub_num=None, **kwargs):
        df = self.get_df_feat(**kwargs)
        df_samples = self.get_df_samples(True, **kwargs)
        clust = self.get_cluster(cl_name, n_clust)
        samples_clust = self.get_samples_cluster(cl_name, n_clust)
        col1 = df.columns[iter_cols[0]]
        col2 = df.columns[iter_cols[1]]

        c_tot = [Constants.colors[i] for i in clust]
        c_samp = [Constants.colors[i] for i in samples_clust]
        if sub_num:
            plt.subplot(sub_num) 
        plt.scatter(x=df[col1],y=df[col2],c=c_tot) 
        plt.scatter(x=df_samples[col1],y=df_samples[col2], lw=1, 
            facecolor=c_samp,marker='D',edgecolors='black') 
        if not centers is None:
            plt.scatter(centers[:,iter_cols[0]],centers[:,iter_cols[1]], lw=1,
                facecolor=Constants.colors[0:n],marker='X',edgecolors='k',s=150)

        plt.xlabel(col1)
        plt.ylabel(col2)

    # }}}

    # {{{ Private helpers
    @classmethod
    def __combine_name_n(cls, name, n):
        return str(name) + '_' + str(n)

    @classmethod
    def __select_plot_cols(cls, ncol):
        if ncol==3:
            iter_cols = [[0,1],[2,1],[0,2],[2,0]]
        elif ncol==4:
            iter_cols = [[0,1],[3,1],[0,2],[3,2]]
        elif ncol==5:
            iter_cols = [[0,1],[2,3],[0,4],[2,4]]
        elif ncol==6:
            iter_cols = [[0,1],[2,3],[4,5],[2,0]]
        return iter_cols


    @classmethod
    def __cluster_mapping(cls, cl1, cl2):
        n1 = max(cl1)-min(cl1)+1
        n2 = max(cl2)-min(cl2)+1
        if n1 != n2:
            raise Exception("Different number of clusters")
        clmap = np.zeros([n1,2])
        for i in range(0,n1):
            iSum = 0 
            for j in range(0,n1):
                nComp = np.logical_and(cl1==i,cl2==j)
                if sum(nComp) > iSum:
                    iSelected = j 
                    iSum = sum(nComp)
            if iSum < 0.5 * sum(cl1==i):
                raise Exception("Can't determine cluster mapping")
            clmap[i,0]=i
            clmap[i,1]=iSelected
        clmap = clmap.astype(int)
        return clmap
    # }}}

if __name__=='__main__':
    prod = Prodotti.get_df_group_prod()
    CA = ClAnalyzer("Kmeans", prod)
    samples = ['P0011AN','P0018AN','P0080AB'] 
    features = ['Ratio','nProv','NordSud']
    CA.features = features
    CA.set_samples(samples,'ProductId')
    CA.print_relevance()
    CA.print_outliers()
    embed()

