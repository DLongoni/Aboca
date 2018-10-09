#!/usr/bin/env python

# {{{ Import
import numpy as np
import itertools
import seaborn as sns
import logging
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from Utils import Constants
from IPython import embed
from functools import lru_cache
from DA import Prodotti
# }}}

sns.set()


# The idea is to make it independent on the clusterer
class ClAnalyzer:
    BASE_DF = 'base'

    def __init__(self, df):
        self._df_dic = {self.BASE_DF: df}
        self._clust_dic = {}
        self._centers_dic = {}
        self._samples_column = 'index'
        self._samples_ids = None
        self._features = None

    # {{{ Properties
    # TODO: think about this. Features are here just for convenience, but
    # all the methods that use features in practice allow for other
    # features to be given...
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
            raise KeyError('Dataframe [{0}] already in '
                           'dictionary'.format(name))
        if (not (df.index == self._df_dic[self.BASE_DF].index).all()):
            raise ValueError("The original and scaled df indices "
                             "must be the same")
        self._df_dic[name] = df

    def set_samples(self, samples_labels, df_column=None):
        if df_column:
            if df_column not in self.get_df().columns:
                raise Exception("Column [{0}] not in df".format(df_column))
            self._samples_column = df_column
            basedf = self._df_dic[self.BASE_DF]
            self.samples_ids = basedf[basedf[df_column].isin(
                samples_labels)].index
        else:
            self._samples_column = 'index'
            self.samples_ids = samples_labels

    def add_cluster(self, clust_arr, clust_name, n_clust, **kwargs):
        key = self.__combine_name_n(clust_name, n_clust)
        if key in self._clust_dic:
            raise ValueError("Cluster with name [{0}] and n [{1}] already in "
                             "the dictionary".format(clust_name, n_clust))
        if len(clust_arr) != self.n:
            raise ValueError("Cluster array have [{0}] elements, but the "
                             "dataset have [{1}]".format(
                                 len(clust_arr), self.n))
        new_c = _Cluster(clust_name, n_clust=n_clust,
                         labels=clust_arr, **kwargs)
        self._clust_dic[key] = new_c
    # }}}

    # {{{ Getters
    @lru_cache(maxsize=10)
    def get_df(self, **kwargs):
        df_name = kwargs.get("df_name", self.BASE_DF)
        feat_cols = kwargs.get("feat_cols", False)
        cols = list(kwargs.get("cols", []))
        if df_name not in self._df_dic:
            raise KeyError('Dataframe [{0}] not set'.format(df_name))
        if feat_cols:
            if self._features is None:
                raise Exception("Set features first")
            return self._df_dic[df_name][self._features]
        elif len(cols) > 0:
            return self._df_dic[df_name][cols]
        else:
            return self._df_dic[df_name]

    @lru_cache(maxsize=10)
    def get_df_samples(self, **kwargs):
        if self._samples_ids is None:
            raise Exception("Samples ids are not set")
        return self.get_df(**kwargs).loc[self._samples_ids]

    @lru_cache(maxsize=10)
    def get_cluster(self, clust_name, n_clust):
        key = self.__combine_name_n(clust_name, n_clust)
        if key not in self._clust_dic:
            raise KeyError("Cluster with name [{0}] and n [{1}] not in the "
                           "dictionary".format(clust_name, n_clust))
        return self._clust_dic[key]

    @lru_cache(maxsize=10)
    def get_samples_labels(self, clust_name, n_clust):
        cl_tot = self.get_cluster(clust_name, n_clust).labels
        return cl_tot[self._samples_ids]

    def get_clust_centers(self, labels, clust_num=-1, **kwargs):
        df = self.get_df(**kwargs)
        # useful when there is an empty cluster
        n_clust = kwargs.get("n_clust", max(labels)+1)
        centers = np.zeros([n_clust, len(df.columns)])
        for iC in range(0, n_clust):
            df_clust = df.loc[labels == iC]
            centers[iC, :] = df_clust.mean().values

        if clust_num == -1:
            return centers
        else:
            return centers[clust_num, :]
    # }}}

    # {{{ Features
    def print_relevance(self, features=None, **kwargs):
        df = self.get_df(**kwargs, feat_cols=True)
        if features is not None:
            cols = self._features
        else:
            if self._features is None:
                raise Exception("Set features first")
            cols = self._features
        print("Relevance analysis for features [{0}]".format(' '.join(cols)))
        for col in cols:
            new_data = df.drop(col, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(
                new_data, df[col], test_size=0.25, random_state=1)
            regressor = DecisionTreeRegressor(random_state=1)
            regressor.fit(X_train, y_train)
            score = regressor.score(X_test, y_test)
            print('Variable ', col, ' predictability: ', score)

    # mode = "iqr" or "2perc"
    def print_outliers(self, mode='iqr', features=None, **kwargs):
        df_out = self.get_df(**kwargs, feat_cols=True)
        df_plot = self.get_df()
        if features is not None:
            cols = self._features
        else:
            if self._features is None:
                raise Exception("Set features first")
            cols = self._features
        print("Outlier analysis for features [{0}]".format(' '.join(cols)))
        for col in cols:
            if mode == 'iqr':
                Q1 = np.percentile(df_out[col], 25)
                Q3 = np.percentile(df_out[col], 75)
                step = (Q3 - Q1) * 1.5
                print("Data points considered step-outliers for the "
                      "col '{}':".format(col))
                print(df_plot[~((df_out[col] >= Q1 - step) &
                                (df_out[col] <= Q3 + step))])
            elif mode == '2perc':
                Qmin = np.percentile(df_out[col], 2)
                Qmax = np.percentile(df_out[col], 98)
                print("Data points considered 2-percent outliers for the "
                      "col '{}':".format(col))
                print(df_plot[~((df_out[col] >= Qmin) &
                                (df_out[col] <= Qmax))])
            else:
                raise ValueError('Mode should be either iqr or 2perc')
    # }}}

    # {{{ Print clusters
    def print_scores(self, **kwargs):
        df_score = self.get_df(**kwargs)
        for key, val in self._clust_dic.items():
            s = silhouette_score(df_score, val.labels)
            c = calinski_harabaz_score(df_score, val.labels)
            print("Scores for cluster [{0}]".format(key))
            print("\t silhouette: [{0:.4f}]".format(s))
            print("\t calinski  : [{0:.4f}]".format(c))

    def describe_clusters(self, clust_name, n_clust, **kwargs):
        clust   = self.get_cluster(clust_name, n_clust)
        df      = self.get_df(**kwargs)
        c_range = range(0, n_clust)
        print("*** Describing clustering [{0}] with [{1}] clusters ***"
              .format(clust_name, n_clust))
        for ic in c_range:
            print("*** Custer labeled [{0}]".format(str(ic)))
            i_df = df.iloc[clust.labels == ic]
            print(i_df.describe())

    def print_sample_clusters(self, clust_name, n_clust, **kwargs):
        samples_labels = self.get_samples_labels(clust_name, n_clust)
        s_range        = range(0, len(self._samples_ids))
        print("*** Samples ***")
        print(self.get_df_samples(**kwargs))
        for i_s in s_range:
            print("Sample [{0}] is in cluster [{1}]"
                  .format(self._samples_ids[i_s], samples_labels[i_s]))

    def print_cluster_diff(self, n_clust, name1, name2,
                           df_name=None, features=None):
        cl1     = self.get_cluster(name1, n_clust).labels
        cl2     = self.get_cluster(name2, n_clust).labels
        clmap   = self.__cluster_mapping(cl1, cl2)
        df_orig = self.get_df()
        print("*** Comparing cluster [{0}] and [{1}] for [{2}] clusters ***"
              .format(name1, name2, n_clust))
        print("Cluster mapping: [\n{0}\n]".format(clmap))
        if df_name is not None:
            if features is None:
                df_score = self.get_df(df_name=df_name, feat_cols=True)
            else:
                df_score = self.get_df(df_name=df_name, cols=features)
            s1 = silhouette_score(df_score, cl1)
            c1 = calinski_harabaz_score(df_score, cl1)
            s2 = silhouette_score(df_score, cl2)
            c2 = calinski_harabaz_score(df_score, cl2)
            print("Silhouette: [{0}]:[{1:.4f}]; [{2}]:[{3:.4f}]"
                  .format(name1, s1, name2, s2))
            print("Calinski:   [{0}]:[{1:.4f}]; [{2}]:[{3:.4f}]"
                  .format(name1, c1, name2, c2))
        for i in range(0, len(clmap)):
            print("***** Comparing cluster [{0}-{1}] with cluster [{2}-{3}]".
                  format(i, name1, clmap[i], name2))
            print("Describing cluster [{0}-{1}]".format(i, name1))
            print(df_orig.iloc[cl1 == i].describe())
            print("Describing cluster [{0}-{1}]".format(clmap[1], name2))
            print(df_orig.iloc[cl2 == clmap[1]].describe())
            cl1notcl2 = np.logical_and(cl1 == i,
                                       np.logical_not(cl2 == clmap[i]))
            if cl1notcl2.any():
                print("Points that are in [{0}] but not in [{1}]"
                      .format(name1, name2))
                print(df_orig.iloc[cl1notcl2])
            else:
                print("All points that are in [{0}] are in [{1}]"
                      .format(name1, name2))

            cl2notcl1 = np.logical_and(cl2 == clmap[i],
                                       np.logical_not(cl1 == i))
            if cl2notcl1.any():
                print("Points that are in [{0}] but not in [{1}]"
                      .format(name2, name1))
                print(df_orig.iloc[cl2notcl1])
            else:
                print("All points that are in [{0}] are in [{1}]"
                      .format(name2, name1))
    # }}}

    # {{{ Plot clusters
    def plot_cluster_diff(self, n_clust, name1, name2, save=False, **kwargs):
        df_name = kwargs.get("df_name", self.BASE_DF)
        n_clust2 = kwargs.get("n_clust2", n_clust)
        cl1     = self.get_cluster(name1, n_clust).labels
        cl2     = self.get_cluster(name2, n_clust2).labels
        print("*** Comparing cluster [{0}] for [{2}] clusters and [{1}] "
              "for [{3}] clusters ***"
              .format(name1, name2, n_clust, n_clust2))
        clmap = []
        if n_clust == n_clust2:
            clmap = self.__cluster_mapping(cl1, cl2)
            print("Cluster mapping: [\n{0}\n]".format(clmap))

        ncol = len(self.get_df(**kwargs).columns)
        if ncol == 2:
            f, axarr = plt.subplots(1, 2)
            self.add_cluster_plot(name1, n_clust, [0, 1], axarr[0],
                                  cluster_map=clmap, **kwargs)
            self.add_cluster_plot(name2, n_clust2, [0, 1], axarr[1], **kwargs)
            axarr[0].set_title(name1, fontweight='bold', fontsize=16)
            axarr[1].set_title(name2, fontweight='bold', fontsize=16)
        else:
            f, axarr = plt.subplots(2, 2)
            iter_cols = self.__select_plot_cols(ncol)
            self.add_cluster_plot(name1, n_clust, iter_cols[0], axarr[0, 0],
                                  cluster_map=clmap, **kwargs)
            self.add_cluster_plot(name2, n_clust2, iter_cols[0], axarr[0, 1],
                                  **kwargs)
            self.add_cluster_plot(name1, n_clust, iter_cols[1], axarr[1, 0],
                                  cluster_map=clmap, **kwargs)
            self.add_cluster_plot(name2, n_clust2, iter_cols[1], axarr[1, 1],
                                  **kwargs)
            axarr[0, 0].set_title(name1+str(n_clust), fontweight='bold',
                                  fontsize=16)
            axarr[0, 1].set_title(name2+str(n_clust2), fontweight='bold',
                                  fontsize=16)

        plotname = name1 + str(n_clust) + "_vs_" + name2 + "_" + \
            str(n_clust2) + "_" + df_name
        f.suptitle(plotname)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if save:
            f.savefig(Constants.pic_path+plotname+'.png')
        plt.show()

    def plot_cluster(self, cl_name, n_clust, save=False, **kwargs):
        df_name = kwargs.get("df_name", self.BASE_DF)
        plotname = cl_name + "_" + str(n_clust) + "_" + df_name

        ncol = len(self.get_df(**kwargs).columns)
        if ncol == 2:
            f, ax = plt.subplots()
            self.add_cluster_plot(cl_name, n_clust, [0, 1], ax, **kwargs)
        else:
            f, axarr = plt.subplots(2, 2)
            iter_cols = self.__select_plot_cols(ncol)
            for ax_ids, it_cols in zip(itertools.product([0, 1], [0, 1]),
                                       iter_cols):
                self.add_cluster_plot(cl_name, n_clust, it_cols,
                                      axarr[ax_ids], **kwargs)
        f.suptitle(plotname)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if save:
            f.savefig(Constants.pic_path+plotname+'.png')
        plt.show()

    def add_cluster_plot(self, cl_name, n_clust, iter_cols, ax, **kwargs):
        cluster_map = kwargs.pop('cluster_map', [])
        samples_label_col = kwargs.pop('samples_label_col', "")
        df = self.get_df(**kwargs)
        df_samples = self.get_df_samples(**kwargs)
        clust = self.get_cluster(cl_name, n_clust).labels
        samples_labels = self.get_samples_labels(cl_name, n_clust)
        if len(cluster_map) > 0:
            clust = np.asarray([cluster_map[i_cl] for i_cl in clust])
            samples_labels = np.asarray([cluster_map[i_cl]
                                         for i_cl in samples_labels])
        col1 = df.columns[iter_cols[0]]
        col2 = df.columns[iter_cols[1]]

        c_tot = [Constants.colors[i] for i in clust]
        c_samp = [Constants.colors[i] for i in samples_labels]
        ax.scatter(x=df[col1], y=df[col2], c=c_tot)
        ax.scatter(x=df_samples[col1], y=df_samples[col2], lw=1,
                   facecolor=c_samp, marker='D', edgecolors='black')
        if samples_label_col != "":
            df_samples_labels = self.get_df_samples(cols=(samples_label_col,))
            for x, y, l in zip(df_samples[col1], df_samples[col2],
                               df_samples_labels.values[:, 0]):
                compr_l = l[0:20] if len(l) < 20 else l[0:17] + '...'
                ax.text(x, y, compr_l, fontsize=10, ha='center')

        centers = self.get_clust_centers(clust, n_clust=n_clust, **kwargs)
        ax.scatter(centers[:, iter_cols[0]], centers[:, iter_cols[1]], lw=1,
                   facecolor=Constants.colors[0:n_clust], marker='X',
                   edgecolors='k', s=150)

        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
    # }}}

    # {{{ Private helpers
    @classmethod
    def __combine_name_n(cls, name, n):
        return str(name) + '_' + str(n)

    @classmethod
    def __select_plot_cols(cls, ncol):
        if ncol == 3:
            iter_cols = [[0, 1], [2, 1], [0, 2], [2, 0]]
        elif ncol == 4:
            iter_cols = [[0, 1], [3, 1], [0, 2], [3, 2]]
        elif ncol == 5:
            iter_cols = [[0, 1], [2, 3], [0, 4], [2, 4]]
        elif ncol == 6:
            iter_cols = [[0, 1], [2, 3], [4, 5], [2, 0]]
        return iter_cols

    @classmethod
    def __cluster_mapping(cls, cl1, cl2):
        n1 = max(cl1)-min(cl1)+1
        n2 = max(cl2)-min(cl2)+1
        clmap = []
        if n1 != n2:
            raise Exception("Different number of clusters")
        for i in range(0, n1):
            iSum = 0
            for j in range(0, n1):
                if j not in clmap:
                    nComp = np.logical_and(cl1 == i, cl2 == j)
                    if sum(nComp) > iSum:
                        iSelected = j
                        iSum = sum(nComp)
            if iSum < 0.5 * sum(cl1 == i):
                logging.warn("Can't determine cluster mapping")
                return []
            clmap.append(iSelected)
        return clmap
    # }}}


# {{{ Cluster class
class _Cluster:
    def __init__(self, name, **kwargs):
        self._name    = name
        self._labels  = kwargs.get("labels", [])
        self._centers = kwargs.get("centers", [])
        self._dataset = kwargs.get("dataset", "")
        self._n_clust = kwargs.get("n_clust", "")

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def centers(self):
        return self._centers

    @centers.setter
    def centers(self, value):
        self._centers = value

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @property
    def n_clust(self):
        return self._n_clust

    @n_clust.setter
    def n_clust(self, value):
        self._n_clust = value

    @property
    def name(self):
        return self._name
# }}}


if __name__ == '__main__':
    prod = Prodotti.get_df_group_prod()
    CA = ClAnalyzer("Kmeans", prod)
    samples = ['P0011AN', 'P0018AN', 'P0080AB']
    features = ['Ratio', 'nProv', 'NordSud']
    CA.features = features
    CA.set_samples(samples, 'ProductId')
    CA.print_relevance()
    CA.print_outliers()
    embed()
