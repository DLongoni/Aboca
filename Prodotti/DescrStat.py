# {{{ Import
import seaborn as sns  # NOQA
from matplotlib import pyplot as plt  # NOQA
from matplotlib import ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
from DA import Prodotti
from DA import Avatar
from Utils import Constants
from IPython import embed  # NOQA
# }}}


sns.set()
sns.set_style('ticks')

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

# {{{ Grafico prodotti piu frequenti
prod_count = Prodotti.get_df_group_prod(data)
piu_frequenti = prod_count[prod_count.nTot >=
                           prod_count.quantile(0.75).nTot]
data.drop(['ProductReplaced', 'CreationTime', 'Id'], axis=1, inplace=True)


def freq_hist(df):
    # sns.set_style('darkgrid')
    plt.ion()
    f = plt.figure()
    ax = f.add_subplot(111)
    dfp = df.sort_values('nTot')
    ax.barh(dfp.ProductId, dfp.Ratio*100)
    for i, (i_name, i_tot) in enumerate(zip(dfp.Name, dfp.nTot)):

        i_lbl = "{0} - {1}".format(i_tot, i_name)
        ax.text(1, i, i_lbl, color="w", va="center", size=16)

    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.tick_params(labelsize=16)
    ax.set_xticks([0, 50, 75, 100])
    xgrid = ax.xaxis.get_gridlines()
    xgrid[1].set_color('r')
    xgrid[1].set_ls('--')
    xgrid[1].set_lw(2)
    xgrid[2].set_color('r')
    xgrid[2].set_ls('--')
    xgrid[2].set_lw(2)
    ax.xaxis.grid(True)

    ax.yaxis.grid(False)
    ax.set_yticks([])
    ax.set_xlim([0, 100])
    ax.set_axisbelow(False)
    ax.set_title('I prodotti più comuni sono consigliati correttamente?',
                 size=20)
    ax.set_xlabel(r'% di consigli corretti', size=18)
    ax.set_ylabel('Numero di consigli', size=18)
    f.tight_layout()
    plt.show()


if 0:
    freq_hist(piu_frequenti)
# }}}

# {{{ Prodotti più rari
if 1:
    df_rari = Prodotti.get_df_group_prod(include_rare=True).sort_values('nTot')
    p_rari = df_rari[df_rari.nTot <= df_rari.nTot.quantile(.1)]
    print('*** Prodotti consigliati rarmente ***')
    print(p_rari[['Name', 'nTot']])
    csv_rari = p_rari[['Name', 'nTot']].to_csv()
# }}}


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


# {{{ Analisi temporale per prodotto e utente
id_test = 'P0011AN'  # Il piu frequente Flora Intestinale Bustine
gr_col = 'YearMonth'
dt_single = data[data.ProductId == id_test]
# for id_test in piu_frequenti.ProductId:
# for id_test in [id_test]:
#     dt_single = data[data.ProductId == id_test]
#     ym_single = rwcount(dt_single, gr_col, 'UserId')
#     ym_single.plot(y='Ratio', kind='bar', colors='b')
#     plt.show()
#     input('press enter')

# Per user - per product (utente con il massimo numero di raccomandazioni
# sul prodotto)
# Non ha senso ragionare sui creation time intraday perchè i dati vengono
# acquisiti in batch e di conseguenza il
# Creation time è identico per "sessione"
# La cosa interessante qua è che vedi quali farmacisti consigliano un botto
# questo farmaco
recomm_user = dt_single.UserId.value_counts()
user_max = recomm_user[recomm_user == recomm_user.max()].index[0]

days_recomm = data_tot.groupby('UserId')['YMD'].nunique()
days_recomm = days_recomm.sort_values(ascending=False)
user_max = days_recomm.index[2]
user_hist = data_tot[data_tot.UserId == user_max]

if 0:  # se voglio fare analisi per prodotto
    p_freq = 'P0096AB'  # prodotto consigliato frequentemente da questo
    user_hist = user_hist[user_hist.ProductId == p_freq]

gr_col = 'YMD'
uh_day = rwcount(user_hist, 'YMD', 'UserId')
uh_day = uh_day.fillna(0)  # user history
# }}}

# {{{ Analisi per avatar
av = rwcount(data_tot, 'AvId')
av_top = av[av.TotCount >= av.TotCount.quantile(.80)]
avpce = Avatar.get_avatar_pce()
av_top_pce = pd.merge(av_top, avpce, left_index=True, right_on='AvId')
av_top_pce.drop(['SessionId', 'AvatarId'], axis=1, inplace=True)


def av_freq_hist(df):
    plt.ion()
    f = plt.figure()
    ax = f.add_subplot(111)
    dfp = df.sort_values('TotCount')
    ax.barh(range(0, len(dfp)), dfp.Ratio*100)
    for i, (i_pce) in enumerate(dfp.AvatarPce):
        i_hist = ax.get_children()[i]
        i_hist.set_color(Constants.colors[i_pce])
        i_hist.set_height(0.8)
        i_hist.set_edgecolor('k')
        i_hist.set_linewidth(1)

    l_hand = []
    for i in range(0, 5):
        i_patch = mpatches.Patch(color=Constants.colors[i+1],
                                 label=Avatar.pce_descr(i+1))
        l_hand.append(i_patch)

    ax.legend(handles=l_hand)
    for i, (i_id, i_tot) in enumerate(zip(dfp.AvId, dfp.TotCount)):

        i_lbl = "{0} - Av{1}".format(i_tot, i_id)
        ax.text(1, i-0.1, i_lbl, color="k", va="center", size=11)

    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.tick_params(labelsize=16)
    ax.set_xticks([0, 25, 50, 75, 100])
    xgrid = ax.xaxis.get_gridlines()
    xgrid[1].set_color('k')
    xgrid[1].set_ls('--')
    xgrid[1].set_lw(1)
    xgrid[2].set_color('k')
    xgrid[2].set_ls('--')
    xgrid[2].set_lw(1)
    xgrid[3].set_color('k')
    xgrid[3].set_ls('--')
    xgrid[3].set_lw(1)
    ax.xaxis.grid(True)

    ax.yaxis.grid(False)
    ax.set_yticks([])
    ax.set_xlim([0, 100])
    ax.set_axisbelow(False)
    ax.set_title('Gli avatar più giocati hanno ricevuto prodotti corretti?',
                 size=20)
    ax.set_xlabel(r'% di consigli corretti', size=18)
    ax.set_ylabel('Numero di prodotti consigliati', size=18)
    f.tight_layout()
    plt.show()


# av_bad = av_top[av_top.Ratio<0.15]
av_worst = int(av_top[av_top.Ratio == av_top.Ratio.min()].index.values)
df_av_worst = data_tot[data_tot.AvId == av_worst]
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
    data_tot = data_tot[data_tot.CreationTime < cut_date]
pce = rwcount(data_tot, 'AvatarPce')

pce_t = 5
# analisi per pce, per prodotto
data_rwt = data_tot[data_tot.AvatarPce == pce_t]
spce = rwcount(data_rwt, 'ProductId')
# analisi per pce, per avatar
apce = rwcount(data_rwt, 'AvId')
# }}}

# {{{ Analisi per utente
prov = rwcount(data_tot, 'Regione')

users_per_reg = data_tot.groupby('Regione')['UserId'].nunique().reset_index()
users_per_reg.set_index('Regione', inplace=True)

prov = pd.merge(prov, users_per_reg, left_index=True, right_index=True)
rol = rwcount(data_tot, 'RoleId')
# }}}

embed()
