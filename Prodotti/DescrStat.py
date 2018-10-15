# {{{ Import
import seaborn as sns
from matplotlib import pyplot as plt  # NOQA
from matplotlib import ticker as ticker
import pandas as pd
from DA import Prodotti
from IPython import embed  # NOQA
# }}}


sns.set()

# {{{ Caricamneto dati
data_tot = Prodotti.get_df()
data = data_tot.copy(deep=True)

# Product names
names = data_tot[['ProductId', 'ProductName', 'ProductFormat']]
names = names.groupby('ProductId')['ProductName',
                                   'ProductFormat'].first().reset_index()
names['Name'] = names['ProductName'] + ' ' + names['ProductFormat']
names.drop(['ProductName', 'ProductFormat'], axis=1, inplace=True)
data_tot = pd.merge(data_tot, names)
data_tot.drop(['ProductName', 'ProductFormat'], axis=1, inplace=True)
# }}}

# {{{ Info prodotto - prezzi, frequency, etc...
prod_count = Prodotti.get_df_group_prod(data)
piu_frequenti = prod_count[prod_count.nTot >=
                           prod_count.quantile(0.75).nTot]
meno_frequenti = prod_count[prod_count.nTot <=
                            prod_count.quantile(0.25).nTot]
data.drop(['ProductReplaced', 'CreationTime', 'Id'], axis=1, inplace=True)
# }}}


def freq_hist(df):
    plt.ion()
    f = plt.figure()
    ax = f.add_subplot(111)
    # lbl = df.Name
    # lbl = [x[0:18] + '...' if len(x) > 18 else x for x in df.Name]
    df.sort_values('nTot', inplace=True)
    ax.barh(df.ProductId, df.Ratio*100)
    for i, (i_name, i_tot, i_val) in enumerate(
            zip(df.Name, df.nTot, df.Ratio*100)):

        ax.text(1, i, i_name, color="w", va="center", size=16)
        ax.text(i_val-5, i, i_tot, color="w", va="center", ha='right', size=14)

    # ax.set_xticklabels(lbl, rotation=45, ha='right')
    ax.yaxis.grid(False)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xticks([0, 50, 75, 100])
    ax.set_yticks([])
    ax.set_xlim([0, 100])
    ax.set_axisbelow(False)
    ax.set_title(r'% di consigli corretti', size=20)
    plt.tight_layout()
    plt.show()


def rwcount(df, group, count_col='Id'):
    df_r = df[df.ProductType == 'RightProduct']
    df_gr = df.groupby(group)[count_col].count().reset_index()
    df_gr.set_index(group, inplace=True)
    df_gr.rename(columns={count_col: 'TotCount'}, inplace=True)
    df_gr_r = df_r.groupby(group)[count_col].count().reset_index()
    df_gr_r.set_index(group, inplace=True)
    df_gr_r.rename(columns={count_col: 'RightCount'}, inplace=True)
    df_gr = pd.merge(df_gr, df_gr_r, left_index=True, right_index=True,
                     how='outer')
    df_gr['Ratio'] = df_gr.RightCount/df_gr.TotCount
    return df_gr


# {{{ Analisi singolo prodotto
id_test = 'P0011AN'  # Il piu frequente Flora Intestinale Bustine
gr_col = 'YearMonth'
# for id_test in piu_frequenti.ProductId:
for id_test in [id_test]:
    dt_single = data[data.ProductId == id_test]
    ym_single = rwcount(dt_single, gr_col, 'UserId')
    # ym_single.plot(y='Ratio',kind='bar',colors='b')
    # pyplot.show()
    # input('press enter')

# Per user - per product (utente con il massimo numero di raccomandazioni
# sul prodotto)
# Non ha senso ragionare sui creation time intraday perchè i dati vengono
# acquisiti in batch e di conseguenza il
# Creation time è identico per "sessione"
# La cosa interessante qua è che vedi quali farmacisti consigliano un botto
# questo farmaco
recomm_user = dt_single.UserId.value_counts()
user_max = recomm_user[recomm_user == recomm_user.max()].index[0]
# }}}

# {{{ Analisi su utente
data_rw = data_tot.drop(data_tot.index[
    data_tot.ProductType == 'RecommendedProduct'])
data_rw = data_rw.drop(data_rw.index[data_rw.ProductType == 'SoldProduct'])
days_recomm = data_rw.groupby('UserId')['YMD'].nunique()
days_recomm = days_recomm.sort_values(ascending=False)
user_max = days_recomm.index[2]
user_hist = data_rw[data_rw.UserId == user_max]

if 0:  # se voglio fare analisi per prodotto
    p_freq = 'P0096AB'  # prodotto consigliato frequentemente da questo
    user_hist = user_hist[user_hist.ProductId == p_freq]

gr_col = 'YMD'
uh_day = rwcount(user_hist, 'YMD', 'UserId')
uh_day = uh_day.fillna(0)  # user history
# }}}

# {{{ Analisi per avatar
av = rwcount(data_rw, 'AvId')
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
if 0:  # se volglio fare analisi indietro nel tempo
    cut_date = pd.Timestamp(2018, 1, 1)
    print('Cutting PCE analisys at ', cut_date)
    data_rw = data_rw[data_rw.CreationTime < cut_date]
pce = rwcount(data_rw, 'AvatarPce')

pce_t = 5
# analisi per pce, per prodotto
data_rwt = data_rw[data_rw.AvatarPce == pce_t]
spce = rwcount(data_rwt, 'ProductId')
# analisi per pce, per avatar
apce = rwcount(data_rwt, 'AvId')
# }}}

# {{{ Analisi per utente
prov = rwcount(data_rw, 'Regione')

users_per_reg = data_rw.groupby('Regione')['UserId'].nunique().reset_index()
users_per_reg.set_index('Regione', inplace=True)

prov = pd.merge(prov, users_per_reg, left_index=True, right_index=True)
rol = rwcount(data_rw, 'RoleId')
# }}}

embed()
