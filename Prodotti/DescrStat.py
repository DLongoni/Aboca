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
# }}}

# {{{ Grafico prodotti piu frequenti
prod_count = Prodotti.get_df_group_prod(data)
piu_frequenti = prod_count[prod_count.nTot >=
                           prod_count.quantile(0.75).nTot]
data.drop(['ProductReplaced', 'CreationTime', 'Id'], axis=1, inplace=True)


def freq_hist(df, title=""):
    # sns.set_style('darkgrid')
    if title == "":
        title = 'I prodotti più comuni sono consigliati correttamente?'
    # plt.ion()
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(111)
    dfp = df.sort_values('nTot')
    ax.barh(dfp.ProductId, dfp.Ratio*100)
    for i, (i_name, i_tot) in enumerate(zip(dfp.ProdName, dfp.nTot)):

        i_lbl = "{0} - {1}".format(i_tot, i_name)
        ax.text(1, i, i_lbl, color="k", va="center", size=16)

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
    ax.set_title(title, size=20)
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
    print(p_rari[['ProdName', 'nTot']])
    csv_rari = p_rari[['ProdName', 'nTot']].to_csv()
# }}}


def rwcount(df, group, count_col='Id'):
    df_r = df[df.ProductType == 'RightProduct']
    df_gr = df.groupby(group)[count_col].count().reset_index()
    df_gr.set_index(group, inplace=True)
    df_gr.rename(columns={count_col: 'nTot'}, inplace=True)
    df_gr_r = df_r.groupby(group)[count_col].count().reset_index()
    df_gr_r.set_index(group, inplace=True)
    df_gr_r.rename(columns={count_col: 'RightCount'}, inplace=True)
    df_gr = pd.merge(df_gr, df_gr_r, left_index=True, right_index=True,
                     how='outer')
    df_gr['Ratio'] = df_gr.RightCount/df_gr.nTot
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

dt_userday = data_tot[['UserId', 'YMD']].drop_duplicates()
days_recomm = dt_userday.groupby('UserId')['YMD'].nunique()
days_recomm = days_recomm.sort_values(ascending=False)


def df_user_history(user_id):
    user_hist = data_tot[data_tot.UserId == user_id]
    uh_day = rwcount(user_hist, 'YMD', 'UserId')
    uh_day = uh_day.fillna(0)  # user history
    return uh_day


user_max = days_recomm.index[1]
user_hist = df_user_history(user_max)

if 0:  # se voglio fare analisi per prodotto
    p_freq = 'P0096AB'  # prodotto consigliato frequentemente da questo
    user_hist = user_hist[user_hist.ProductId == p_freq]
# }}}

# {{{ Analisi per Avatar
av = rwcount(data_tot, 'AvSessId')
av_top = av[av.nTot >= av.nTot.quantile(.80)]
avpce = Avatar.get_avatar_pce()
av_top_pce = pd.merge(av_top, avpce, left_index=True, right_on='AvSessId')
# av_top_pce.drop(['SessionId', 'AvatarId'], axis=1, inplace=True)


def av_freq_hist(df, title="", legend=True):
    # plt.ion()
    if title == "":
        title = 'Gli avatar più giocati hanno ricevuto prodotti corretti?'
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(111)
    dfp = df.sort_values('nTot')
    ax.barh(range(0, len(dfp)), dfp.Ratio*100)
    for i, (i_pce) in enumerate(dfp.AvatarPce):
        i_hist = ax.get_children()[i]
        i_hist.set_color(Constants.colors[i_pce])
        i_hist.set_height(0.8)
        i_hist.set_edgecolor('k')
        i_hist.set_linewidth(1)

    if legend:
        l_hand = []
        for i in range(0, 5):
            i_patch = mpatches.Patch(color=Constants.colors[i+1],
                                     label=Avatar.pce_descr(i+1))
            l_hand.append(i_patch)

        ax.legend(handles=l_hand)
    for i, (i_name, i_tot, i_sess) in enumerate(
            zip(dfp.AvName, dfp.nTot, dfp.SessionId)):

        i_lbl = "{0} - {1} - {2}".format(i_tot, i_name, i_sess)
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
    ax.set_title(title, size=20)
    ax.set_xlabel(r'% di consigli corretti', size=18)
    ax.set_ylabel('Numero di prodotti consigliati', size=18)
    f.tight_layout()
    plt.show()


# av_bad = av_top[av_top.Ratio<0.15]
av_worst = int(av_top[av_top.Ratio == av_top.Ratio.min()].index.values)


def worst_prod_breakdown(av_worst):
    df_av_worst = data_tot[data_tot.AvSessId == av_worst]
    # prodotti più frequentemente consigliati sbagliati a questo avatar
    df_worst_rw = rwcount(df_av_worst, 'ProductId')
    df_worst_rw = df_worst_rw.nlargest(10, 'nTot').reset_index().fillna(0)
    df_worst_rw = Prodotti.add_name(df_worst_rw)
    return df_worst_rw


# Se da questa stampa ho tassi di correttezza <> {0,1} allora c'è qualcosa
# che non va perchè significa che ho un prodotto allo stesso tempo
# giusto e sbagliato su uno stesso avatar
if 0:
    for i in av_top_pce.AvSessId:
        print("****** [{0}]".format(i))
        print(worst_prod_breakdown(i))

df_worst_rw = worst_prod_breakdown(av_worst)

print("*** Prodotti consigliati erroneamente all'avatar [{0}] ***".format(
    str(av_worst)))
print(df_worst_rw)
# }}}

# {{{ Analisi per PCE
pce = rwcount(data_tot, 'AvatarPce')

# analisi per pce, per avatar
all_pce = data_tot.AvatarPce.unique()
all_pce.sort()
for i_pce in all_pce:
    data_rwt = data_tot[data_tot.AvatarPce == i_pce]
    apce = rwcount(data_rwt, 'AvSessId')
    apce = apce[apce.nTot > apce.quantile(0.8).nTot].reset_index()
    apce = pd.merge(apce, avpce)

    i_tit = "Avatar più giocati per il PCE {0}".format(
        Avatar.pce_descr(i_pce))
    # print(apce[apce.Ratio == apce.Ratio.min()])
    av_freq_hist(apce, i_tit, False)

# analisi per pce, per prodotto
if 0:
    for i_pce in all_pce:
        data_rwt = data_tot[data_tot.AvatarPce == i_pce]
        apce = rwcount(data_rwt, 'ProductId')
        apce = apce[apce.nTot > apce.quantile(0.8).nTot].reset_index()
        apce = Prodotti.add_name(apce)

        i_tit = "I prodotti più consigliati per il PCE {0}".format(
            Avatar.pce_descr(i_pce))
        # print(apce[apce.Ratio == apce.Ratio.min()])
        freq_hist(apce, i_tit)
        # plt.show()
# }}}

# {{{ Analisi geografica
# TODO: grafico bar o barh prov con colormap ratio e ntot lunghezza barre
prov = rwcount(data_tot, 'Regione')

users_per_reg = data_tot.groupby('Regione')['UserId'].nunique().reset_index()
users_per_reg.set_index('Regione', inplace=True)

prov = pd.merge(prov, users_per_reg, left_index=True, right_index=True)
rol = rwcount(data_tot, 'RoleId')
# }}}

embed()
