#!/usr/bin/env python

# {{{ Import
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D, axes3d

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import Others.Renders as rd
import pandas as pd
import AbocaUsers
import AbocaAvatar
import Utils
# }}}

# {{{ Caricamento Dati
df = pd.read_csv('./Dataset/Dumps/out_VrAvatarProducto.csv',sep='|')
df = AbocaAvatar.merge_avatar(df)
df.drop(['TenantId','DeletionTime','LastModificationTime','SessionId','AvatarId',
    'LastModifierUserId','CreatorUserId','IsDeleted','DeleterUserId','ProductSequence',
    'ProductPce'],axis=1,inplace=True)
df = df.drop(df.index[df.ProductType == 'RecommendedProduct'])
df = df.drop(df.index[df.ProductType == 'SoldProduct'])
df.CreationTime = pd.to_datetime(df.CreationTime, dayfirst = True)
df = Utils.filter_date(df,'CreationTime')
df = AbocaUsers.merge_users_clean(df)
# }}}

# {{{ Aggregazione colonne dataframe
# Names 
prod = df[['ProductId','ProductName','ProductFormat']]
prod = prod.groupby('ProductId')['ProductName','ProductFormat'].first().reset_index()
prod['Name'] = prod['ProductName'] + ' ' + prod['ProductFormat']
prod.drop(['ProductName','ProductFormat'],axis=1,inplace=True)
# prod.set_index('ProductId',true)

n_users = df.groupby('ProductId')['UserId'].nunique().reset_index()
n_users.rename(columns = {'UserId': 'nUsers'}, inplace = True)

n_farma = df.groupby('ProductId')['FarmaId'].nunique().reset_index()
n_farma.rename(columns = {'FarmaId': 'nFarma'}, inplace = True)

n_tot = df.groupby('ProductId')['Id'].count().reset_index()
n_tot.rename(columns = {'Id': 'nTot'}, inplace = True)

df_r = df[df.ProductType == 'RightProduct']
n_r = df_r.groupby('ProductId')['Id'].count().reset_index()
n_r.rename(columns = {'Id': 'nRight'}, inplace = True)

lat_m = df.groupby('ProductId')['Latitudine'].mean().reset_index()
lat_m.Latitudine = lat_m.Latitudine - 42 # Ipotetica latitudine di centro italia
lat_m.rename(columns = {'Latitudine': 'NordSud'}, inplace = True)

lat_v = df.groupby('ProductId')['Latitudine'].var().reset_index()
lat_v = lat_v.fillna(0)
lat_v.rename(columns = {'Latitudine': 'LatVar'}, inplace = True)

n_prov = df.groupby('ProductId')['ProvId'].nunique().reset_index()
n_prov.rename(columns = {'ProvId': 'nProv'}, inplace = True)

n_reg = df.groupby('ProductId')['Regione'].nunique().reset_index()
n_reg.rename(columns = {'Regione': 'nReg'}, inplace = True)

prod=pd.merge(prod,n_users)
prod=pd.merge(prod,n_farma)
prod=pd.merge(prod,n_tot)
prod=pd.merge(prod,n_r)
prod=pd.merge(prod,lat_m)
prod=pd.merge(prod,lat_v)
prod=pd.merge(prod,n_prov)
prod=pd.merge(prod,n_reg)
prod['Ratio']=prod.nRight/prod.nTot
# TODO: eliminare questi ratio perchè secondo me non hanno molta ragione di esistere
prod['UserRatio']=prod.nTot/prod.nUsers
prod['GeoRatio']=prod.nTot/prod.nProv
# pd.plotting.scatter_matrix(prod,diagonal = 'kde')
# pyplot.show()
# nFarma e nUsers sono sostituibili
prod=prod[prod.nTot>2] # seleziono quelli su cui ha senso fare un'analisi? corretto o no?

prod_proc=prod.copy(deep=True)
prod_proc.nUsers=scale(np.log(prod.nUsers))
prod_proc.nFarma=scale(np.log(prod.nFarma))
prod_proc.nTot=scale(np.log(prod.nTot))
prod_proc.nRight=scale(np.log(prod.nRight))
prod_proc.NordSud=scale(prod.NordSud)
prod_proc.LatVar=scale(prod.LatVar)
prod_proc.nProv=scale(np.log(prod.nProv))
prod_proc.nReg=scale(np.log(prod.nReg))
prod_proc.Ratio=scale(prod.Ratio)
prod_proc.UserRatio=scale(np.log(prod.UserRatio))
prod_proc.GeoRatio=scale(np.log(prod.GeoRatio))
# pd.plotting.scatter_matrix(prod_proc,diagonal = 'kde')
# pyplot.show()
# }}}

# {{{ Feature Relevance
# droppo in base alle osservazioni risultanti del ciclo successivo
prod_proc.drop(['nFarma','nRight','nUsers','GeoRatio','LatVar','nReg','nTot'],axis=1,inplace=True)

feats = prod_proc.keys()
feats = feats.drop(['ProductId','Name'])

prod_feat = prod_proc.drop(['ProductId','Name'],axis=1)
for feature in feats:
    new_data = prod_feat.drop(feature, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(new_data, prod_proc[feature], test_size=0.25, random_state=1)
    regressor = DecisionTreeRegressor(random_state=1)
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print('Variable ', feature,' predictability: ', score)
# }}}

# {{{ Outliers
if 0:
    for feature in feats:
        if False: # seleziono tipo di analisi outliers
            Q1 = np.percentile(prod_proc[feature], 25)
            Q3 = np.percentile(prod_proc[feature], 75)
            step = (Q3 - Q1) * 1.5
            print("Data points considered step-outliers for the feature '{}':".format(feature))
            display(prod[~((prod_proc[feature] >= Q1 - step) & (prod_proc[feature] <= Q3 + step))])
        else:
            Qmin = np.percentile(prod_proc[feature], 2)
            Qmax = np.percentile(prod_proc[feature], 98)
            print("Data points considered 2-percent outliers for the feature '{}':".format(feature))
            display(prod[~((prod_proc[feature] >= Qmin) & (prod_proc[feature] <= Qmax))])

# }}}

# {{{ PCA
# 1: prodotto molto consigliato
# 2: prodotto consigliato a nord
# 3: esempio consigliato correttamente e numeroso
esempi = ['P0011AN','P0018AN','P0080AB'] 
esempi_orig = prod[prod.ProductId.isin(esempi)]
esempi_proc = prod_proc[prod_proc.ProductId.isin(esempi)].drop(['ProductId','Name'],axis=1)
print('Esempi selezionati')
display(esempi_orig)
pca = PCA(n_components=2)
pca.fit(prod_feat)
esempi_pca = pca.transform(esempi_proc)
pca_results = rd.pca_results(prod_feat, pca)
pyplot.savefig('./ClusterFig/pcadim.png')

print('Esempi trasformati')
display(pd.DataFrame(np.round(esempi_pca, 4), columns = pca_results.index.values))

prod_red = np.round(pca.transform(prod_feat),4)
df_red =pd.DataFrame(prod_red, columns = pca_results.index.values)
df_samples_red =pd.DataFrame(esempi_pca, columns = pca_results.index.values)
# }}}

def add_cluster_plot(df,df_sample,col1,col2,pred_tot,sample_pred,sub_num):
    col_arr = ['b','g','r','c','m','y']
    c_tot = [col_arr[i] for i in pred_tot]
    c_samp = [col_arr[i] for i in sample_pred]
    if sub_num:
        pyplot.subplot(sub_num) 
    pyplot.scatter(x=df[col1],y=df[col2],c=c_tot) 
    pyplot.scatter(x=df_sample[col1],y=df_sample[col2], lw=1, 
        facecolor=c_samp,marker='D',edgecolors='black') 
    pyplot.xlabel(col1)
    pyplot.ylabel(col2)


# {{{ Clustering - dati originali + Kmeans
data_to_fit = prod_feat
samples_to_fit = esempi_proc
if 0:
    for n_clusters in range(4,5):
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        clusterer.fit(data_to_fit)
        preds = clusterer.predict(data_to_fit)
        centers = clusterer.cluster_centers_
        sample_preds = clusterer.predict(samples_to_fit)
        # Calculate the mean silhouette coefficient for the number of clusters chosen
        score = silhouette_score(data_to_fit, preds, random_state=1)
        print("{0} clusters: {1:.4f}".format(n_clusters, score))
        
        if 1:
            add_cluster_plot(prod,esempi_orig,'Ratio','NordSud',preds,sample_preds,221)
            add_cluster_plot(prod,esempi_orig,'UserRatio','nProv',preds,sample_preds,222)
            add_cluster_plot(prod,esempi_orig,'Ratio','nProv',preds,sample_preds,223)
            add_cluster_plot(prod,esempi_orig,'NordSud','UserRatio',preds,sample_preds,224)
            fname = 'ProdOrig' + str(n_clusters) + '.png'
            pyplot.savefig('./ClusterFig/'+fname)
            pyplot.show()
            input('press enter')
# }}}

# {{{ Clustering - PCA + Kmeans
data_to_fit = df_red
samples_to_fit = df_samples_red
if 1:
    for n_clusters in range(4,5):
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        clusterer.fit(data_to_fit)
        preds = clusterer.predict(data_to_fit)
        centers = clusterer.cluster_centers_
        sample_preds = clusterer.predict(samples_to_fit)
        # Calculate the mean silhouette coefficient for the number of clusters chosen
        score = silhouette_score(data_to_fit, preds, random_state=1)
        print("{0} clusters: {1:.4f}".format(n_clusters, score))
        
        if 1:
            pyplot.figure(0)
            fname = 'ProdPCAdomPCA' + str(n_clusters) + '.png'
            add_cluster_plot(df_red,df_samples_red,'Dimension 1','Dimension 2',preds,sample_preds,None)
            pyplot.title(str(n_clusters)+' clust PCA ')
            # pyplot.savefig('./ClusterFig/'+fname)
            pyplot.figure(1)
            add_cluster_plot(prod,esempi_orig,'Ratio','NordSud',preds,sample_preds,221)
            add_cluster_plot(prod,esempi_orig,'UserRatio','nProv',preds,sample_preds,222)
            add_cluster_plot(prod,esempi_orig,'Ratio','nProv',preds,sample_preds,223)
            add_cluster_plot(prod,esempi_orig,'NordSud','UserRatio',preds,sample_preds,224)
            pyplot.title(str(n_clusters)+' clust PCA ')
            fname = 'ProdPCAdomOrig' + str(n_clusters) + '.png'
            # pyplot.savefig('./ClusterFig/'+fname)
            pyplot.show()
            input('press enter')
# }}}
