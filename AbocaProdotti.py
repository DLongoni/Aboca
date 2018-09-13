#!/usr/bin/env python

# {{{ Import
from matplotlib import pyplot
import pandas as pd
import AbocaUsers
import AbocaAvatar
import Utils
# }}}

# {{{ Caricamneto dati
data_tot = pd.read_csv('./Dataset/Dumps/out_VrAvatarProducto.csv',sep='|')
data_tot = AbocaAvatar.merge_avatar(data_tot)
data_tot.drop(['TenantId','DeletionTime','LastModificationTime','SessionId','AvatarId',
    'LastModifierUserId','CreatorUserId','IsDeleted','DeleterUserId','ProductSequence',
    'ProductPce'],axis=1,inplace=True)
data_tot = AbocaUsers.clean_df_userid(data_tot)
data_tot.CreationTime = pd.to_datetime(data_tot.CreationTime, dayfirst = True)
data_tot = Utils.filter_date(data_tot,'CreationTime')
data_tot = Utils.add_aggregate_date(data_tot,'CreationTime')
data = data_tot.copy(deep=True)

# Names 
names = data_tot[['ProductId','ProductName','ProductFormat']]
names = names.groupby('ProductId')['ProductName','ProductFormat'].first().reset_index()
names['Name'] = names['ProductName'] + ' ' + names['ProductFormat']
names.drop(['ProductName','ProductFormat'],axis=1,inplace=True)
data_tot = pd.merge(data_tot,names)
data_tot.drop(['ProductName','ProductFormat'],axis=1,inplace=True)
# }}}

# {{{ Info prodotto - prezzi, frequency, etc...

# Prices - bisogna usare i prezzi con attenzione perchè non sembrano univoci
price = data_tot[['ProductId','ProductPrice']]
price = price.drop(price.index[np.isnan(price.ProductPrice)])
price = price.groupby('ProductId')['ProductPrice'].first().reset_index()

# Right / Wrong
data.drop(['ProductReplaced'],axis=1,inplace=True)
data.drop(['CreationTime'],axis=1,inplace=True)
data.drop(['Id'],axis=1,inplace=True)
data = data.drop(data.index[data.ProductType == 'RecommendedProduct'])
data = data.drop(data.index[data.ProductType == 'SoldProduct'])

id_group = data.groupby('ProductId')['ProductType']
right_ratio = id_group.apply(lambda x: (x == 'RightProduct').sum()) / id_group.count()
right_ratio = right_ratio.reset_index()
right_ratio.rename(columns = {'ProductType': 'RightCount'}, inplace = True)
id_count = id_group.count().reset_index()
id_count.rename(columns = {'ProductType': 'TotCount'}, inplace = True)
prod_count = pd.merge(right_ratio,id_count)
prod_count = pd.merge(prod_count,names)
prod_count = pd.merge(prod_count,price)

piu_frequenti = prod_count[prod_count.TotCount >= prod_count.quantile(0.75).TotCount]
meno_frequenti = prod_count[prod_count.TotCount <= prod_count.quantile(0.25).TotCount]
# }}}

def rwcount(df,group,count_col = 'Id'):
    df_r = df[df.ProductType == 'RightProduct']
    df_gr = df.groupby(group)[count_col].count().reset_index()
    df_gr.set_index(group, inplace=True)
    df_gr.rename(columns = {count_col: 'TotCount'}, inplace = True)
    df_gr_r = df_r.groupby(group)[count_col].count().reset_index()
    df_gr_r.set_index(group, inplace=True)
    df_gr_r.rename(columns = {count_col: 'RightCount'}, inplace = True)
    df_gr = pd.merge(df_gr,df_gr_r,left_index=True,right_index=True,how='outer')
    df_gr['Ratio'] = df_gr.RightCount/df_gr.TotCount
    return df_gr

# {{{ Analisi singolo prodotto
id_test = 'P0011AN' # Il piu frequente Flora Intestinale Bustine
gr_col = 'YearMonth'
# for id_test in piu_frequenti.ProductId:
for id_test in [id_test]:
    dt_single = data[data.ProductId == id_test]
    ym_single = rwcount(dt_single,gr_col,'UserId')
    # ym_single.plot(y='Ratio',kind='bar',colors='b')
    # pyplot.show()
    # input('press enter')

# Per user - per product (utente con il massimo numero di raccomandazioni sul prodotto)
# Non ha senso ragionare sui creation time intraday perchè i dati vengono acquisiti in batch e di conseguenza il
# Creation time è identico per "sessione"
# La cosa interessante qua è che vedi quali farmacisti consigliano un botto questo farmaco
recomm_user = dt_single.UserId.value_counts()
user_max = recomm_user[recomm_user == recomm_user.max()].index[0]  
# }}}

# {{{ Analisi su utente
data_rw = data_tot.drop(data_tot.index[data_tot.ProductType == 'RecommendedProduct'])
data_rw = data_rw.drop(data_rw.index[data_rw.ProductType == 'SoldProduct'])
days_recomm = data_rw.groupby('UserId')['YMD'].nunique()
days_recomm = days_recomm.sort_values(ascending = False)
user_max = days_recomm.index[2]
user_hist = data_rw[data_rw.UserId == user_max]

if 0: # se voglio fare analisi per prodotto
    p_freq = 'P0096AB' # prodotto consigliato frequentemente da questo
    user_hist = user_hist[user_hist.ProductId == p_freq]

gr_col = 'YMD'
uh_day = rwcount(user_hist,'YMD','UserId')
uh_day = uh_day.fillna(0) # user history 
# }}}

# {{{ Analisi per avatar
av = rwcount(data_rw,'AvId')
av_top = av[av.TotCount >= av.TotCount.quantile(.75)]

# av_bad = av_top[av_top.Ratio<0.15]
av_worst = int(av_top[av_top.Ratio == av_top.Ratio.min()].index.values)
df_av_worst = data_rw[data_rw.AvId == av_worst]
# i pochi che hanno azzeccato questo avatar
df_av_w_right = df_av_worst[df_av_worst.ProductType == 'RightProduct']
df_av_w_wrong = df_av_worst[df_av_worst.ProductType == 'WrongProduct']
df_av_w_wrong_prod = df_av_w_wrong.groupby('ProductId')['Id'].count()
# prodotti più frequentemente consigliati sbagliati a questo avatar
worst_prod_av = df_av_w_wrong_prod.nlargest(3)
# }}}

# {{{ Analisi per PCE 
if 0: # se volglio fare analisi indietro nel tempo
    cut_date = pd.Timestamp(2018,1,1)
    print('Cutting PCE analisys at ', cut_date)
    data_rw = data_rw[data_rw.CreationTime < cut_date]
pce = rwcount(data_rw,'AvatarPce')

pce_t = 5
# analisi per pce, per prodotto
data_rwt = data_rw[data_rw.AvatarPce == pce_t]
spce = rwcount(data_rwt,'ProductId')
# analisi per pce, per avatar
apce = rwcount(data_rwt,'AvId')
# }}}

# {{{ Analisi per utente
du_rw = AbocaUsers.merge_users_clean(data_rw)
prov = rwcount(du_rw,'Regione')

users_per_reg = du_rw.groupby('Regione')['UserId'].nunique().reset_index()
users_per_reg.set_index('Regione', inplace=True)

prov = pd.merge(prov,users_per_reg,left_index=True,right_index=True)
rol = rwcount(du_rw,'RoleId')
# }}}
