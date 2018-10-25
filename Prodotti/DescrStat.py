#!/usr/bin/env python
# coding: utf-8

# {{{ Import
import seaborn as sns  # NOQA
import numpy as np
from matplotlib import pyplot as plt  # NOQA
from matplotlib import ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
from DA import Prodotti
from DA import Users
from DA import DataHelper as dh
from DA import CsvLoader as CsvL
from Utils import Constants
from IPython import embed  # NOQA
# }}}

sns.set()
sns.set_style('ticks')

# {{{ Caricamento dati
data_tot = Prodotti.get_df()
data = data_tot.copy(deep=True)
# }}}

# {{{ Grafico prodotti piu frequenti
prod_count = Prodotti.get_df_group_prod(data)
piu_frequenti = prod_count[prod_count.nTot >=
                           prod_count.quantile(0.75).nTot]
data.drop(['ProductReplaced', 'Id'], axis=1, inplace=True)


def freq_hist(df, title=""):
    # sns.set_style('darkgrid')
    if title == "":
        title = 'I prodotti più comuni sono consigliati correttamente?'
    # plt.ion()
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(111)
    dfp = df.sort_values('nTot')
    ax.barh(dfp.ProductId, dfp.Ratio * 100)
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
    df_gr['Ratio'] = df_gr.RightCount / df_gr.nTot
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
# recomm_user = dt_single.UserId.value_counts()
# user_max = recomm_user[recomm_user == recomm_user.max()].index[0]

uhist = data_tot[['UserId', 'AvSessId', 'SessionId', 'YMD']].drop_duplicates()
uhist_g = uhist.groupby('UserId')[['AvSessId', 'SessionId', 'YMD']].nunique()
uhist_g = uhist_g.reset_index()
uhist_g['AvPerDay'] = uhist_g.AvSessId / uhist_g.YMD
uhist_g['SessPerDay'] = uhist_g.SessionId / uhist_g.YMD
uhist_g['AvPerSess'] = uhist_g.AvSessId / uhist_g.SessionId
# uhist_g = uhist_g.sort_values(ascending=False)


def df_user_history(user_id, group='AvSessId'):
    user_hist = data_tot[data_tot.UserId == user_id]
    uh_day = rwcount(user_hist, group, 'UserId')
    uh_day = uh_day.fillna(0)  # user history
    return uh_day


user_max = uhist_g[uhist_g.YMD == uhist_g.YMD.max()].UserId.values[0]
user_hist = df_user_history(user_max)

if 0:  # se voglio fare analisi per prodotto
    p_freq = 'P0096AB'  # prodotto consigliato frequentemente da questo
    user_hist = user_hist[user_hist.ProductId == p_freq]

user_ratio = rwcount(data_tot, 'UserId').reset_index()
u_rw_hist = pd.merge(user_ratio, uhist_g)
# Grafico con user ratio, num giocate tot, ratio giocate/giorno
if 0:
    plt.scatter(u_rw_hist.nTot, u_rw_hist.Ratio, c=u_rw_hist.AvPerDay,
                cmap='viridis')
    plt.colorbar()
    plt.show()

u_most_sess = uhist_g.nlargest(4, 'SessionId').UserId.values

log_ferrari = CsvL.get_web_log(2668)


def count_occurrences(lb, data):
    ret = np.zeros(len(lb) - 1)
    data = data[data >= lb.min()]
    data = sorted(data[data <= lb.max()])
    lb.sort()
    i_data = 0
    i_bucket = 0
    for i_lb in lb[1:]:
        while (i_data < len(data)) and (data[i_data] < i_lb):
            ret[i_bucket] = ret[i_bucket] + 1
            i_data = i_data + 1
        i_bucket = i_bucket + 1
    return ret


def u_sess_start(uid):
    uh = uhist[uhist.UserId == uid]
    ustart = uh.groupby('SessionId').min().reset_index().YMD
    # labels = ustart.dt.strftime('%d/%m/%Y')
    return ustart


def plot_uhist(uid):
    plt.ion()
    i_uh = df_user_history(uid, 'SessionId')
    i_uh_r = i_uh.rolling(10)
    i_sumR = i_uh_r.RightCount.sum()
    i_sumT = i_uh_r.nTot.sum()
    i_movavg = i_sumR / i_sumT
    r_obs = range(0, len(i_movavg))
    r_obs2 = np.arange(-0.7, len(i_movavg) - 0.7, 1)
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(111)
    ax2 = ax.twinx()
    l_hand = []
    l_lab = []
    line_avg, = ax.plot(r_obs, i_movavg, lw=3, zorder=10)
    bar_ratio = ax.bar(r_obs2, i_uh.Ratio, color='orange',
                       alpha=0.5, width=0.7, align='edge', zorder=1)
    bar_ntot = ax2.bar(r_obs, i_uh.nTot, color='green', alpha=0.5, width=0.25,
                       align='edge')
    weblog = CsvL.get_web_log(uid)
    arr_start = u_sess_start(uid)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    l_hand.append(line_avg)
    l_lab.append('Rolling average')
    l_hand.append(bar_ratio)
    l_lab.append('Correttezza')
    l_hand.append(bar_ntot)
    l_lab.append('Tot prodotti')
    if weblog is not None:
        num_web_check = count_occurrences(arr_start.values, weblog)
        xval = np.add(np.where(num_web_check > 0), 0.25).squeeze(axis=0)
        yval = np.ones(len(xval)) * 0.5
        sca_web = ax.scatter(xval, yval, marker='d', edgecolors='k', s=150,
                             lw=1, facecolor='r', zorder=2)
        l_hand.append(sca_web)
        l_lab.append('Sito web')

    ax.legend(tuple(l_hand), tuple(l_lab), fontsize=16)
    ax.set_ylabel('Tasso di correttezza', size=18)
    ax2.set_ylabel('Numero prodotti consigliati', size=18)
    ax.set_xlabel('Sessioni', size=18)
    ax.set_xticks(r_obs)
    ax.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    lbl = arr_start.dt.strftime('%d/%m/%Y')
    ax.set_xticklabels(lbl, rotation=45, ha='right', size=15)

    plt.title("Correttezza di consiglio prodotti nel corso delle sessioni "
              "per {0}".format(Users.get_user_name(uid)), size=25, y=1.02)
    plt.show()


def prod_plot(uid):
    plt.ion()
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    du = data_tot[data_tot.UserId == uid]
    duprod = rwcount(du, 'ProductId').reset_index()
    most_uprod = duprod.nlargest(4, 'nTot').ProductId.values
    for i, (ip, ia) in enumerate(zip(most_uprod, [ax1, ax2, ax3, ax4])):
        idup = du[du.ProductId == ip]
        idup_rw = idup.groupby('SessionId').apply(lambda x: pd.Series(
            {'Ratio': sum(x.ProductType == 'RightProduct') / x.Id.count(),
             'nTot': x.Id.count()})).reset_index()
        ic = Constants.colors[i]
        r_obs2 = range(0, len(idup_rw))
        r_obs = np.arange(-0.7, len(idup_rw) - 0.7, 1)
        ia.bar(r_obs, idup_rw.Ratio, color=ic, width=0.7, align='edge')
        ia2 = ia.twinx()
        ia2.bar(r_obs2, idup_rw.nTot, color='orange', width=0.25, align='edge')
        ia2.set_yticks(range(0, int(idup_rw.nTot.max() + 1)))
        ipname = Prodotti.get_product_name(ip)
        ia.set_title('Progresso per {0}'.format(ipname), size=16)
        ia.set_xticks([])
        ia.tick_params(labelsize=16)

    a = f.axes
    a[0].set_ylabel('Tasso di correttezza', size=18)
    a[2].set_ylabel('Tasso di correttezza', size=18)
    a[5].set_ylabel('Numero prodotti consigliati', size=18,
                    color='orange')
    a[7].set_ylabel('Numero prodotti consigliati', size=18,
                    color='orange')
    a[2].set_xlabel('Sessioni', size=18)
    a[3].set_xlabel('Sessioni', size=18)

    f.suptitle('Utente {0}'.format(Users.get_user_name(uid)), size=18)
    plt.show()


if 0:
    for i_u in u_most_sess:
        plot_uhist(i_u)
# }}}

# {{{ Analisi per Avatar
av = rwcount(data_tot, 'AvSessId')
av_top = av[av.nTot >= av.nTot.quantile(.80)].reset_index()
# avpce = Avatar.get_avatar_pce()
# av_top_pce = pd.merge(av_top, avpce, left_index=True, right_on='AvSessId')
av_top_pce = dh.add_avatar_data(av_top)
# av_top_pce.drop(['SessionId', 'AvatarId'], axis=1, inplace=True)


def av_freq_hist(df, title="", legend=True):
    # plt.ion()
    if title == "":
        title = 'Gli avatar più giocati hanno ricevuto prodotti corretti?'
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(111)
    dfp = df.sort_values('nTot')
    ax.barh(range(0, len(dfp)), dfp.Ratio * 100)
    for i, (i_pce) in enumerate(dfp.AvatarPce):
        i_hist = ax.get_children()[i]
        i_hist.set_color(Constants.colors[i_pce])
        i_hist.set_height(0.8)
        i_hist.set_edgecolor('k')
        i_hist.set_linewidth(1)

    if legend:
        l_hand = []
        for i in range(0, 5):
            i_patch = mpatches.Patch(color=Constants.colors[i + 1],
                                     label=dh.pce_descr(i + 1))
            l_hand.append(i_patch)

        ax.legend(handles=l_hand)
    for i, (i_name, i_tot, i_sess) in enumerate(
            zip(dfp.AvName, dfp.nTot, dfp.SessionId)):

        i_lbl = "{0} - {1} - {2}".format(i_tot, i_name, i_sess)
        ax.text(1, i - 0.1, i_lbl, color="k", va="center", size=11)

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
av_worst = int(av_top[av_top.Ratio == av_top.Ratio.min()].AvSessId)


def worst_prod_breakdown(av_worst):
    df_av_worst = data_tot[data_tot.AvSessId == av_worst]
    # prodotti più frequentemente consigliati sbagliati a questo avatar
    df_worst_rw = rwcount(df_av_worst, 'ProductId')
    df_worst_rw = df_worst_rw.nlargest(10, 'nTot').reset_index().fillna(0)
    df_worst_rw = dh.add_prod_name(df_worst_rw)
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
all_pce = sorted(data_tot.AvatarPce.unique())

# analisi per pce, per avatar
if 0:
    for i_pce in all_pce:
        data_rwt = data_tot[data_tot.AvatarPce == i_pce]
        apce = rwcount(data_rwt, 'AvSessId')
        apce = apce[apce.nTot > apce.quantile(0.8).nTot].reset_index()
        apce = dh.add_avatar_data(apce)
        # apce = pd.merge(apce, avpce)

        i_tit = "Avatar più giocati per il PCE {0}".format(
            dh.pce_descr(i_pce))
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
            dh.pce_descr(i_pce))
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

# {{{ REGION: Analisi num giocate utenti
ymdcount = uhist_g.YMD.value_counts()
avcount = uhist_g.AvSessId.value_counts()
# Il grafico della morte
# plt.scatter(ymdcount.index, ymdcount.values)


if 1:
    for soglia_s in [11, 19]:
        dsess_max = data_tot.groupby('UserId')['SessionId'].max().reset_index()
        urw = rwcount(data_tot, 'UserId')
        sessratio = pd.merge(dsess_max, urw, on='UserId')
        utanti = sessratio[sessratio.SessionId >= soglia_s].UserId.values
        upochi = sessratio[sessratio.SessionId < soglia_s].UserId.values
        dtanti = data_tot[data_tot.UserId.isin(utanti)]
        dpochi = data_tot[data_tot.UserId.isin(upochi)]
        rwta = rwcount(dtanti, 'SessionId').reset_index()
        rwpo = rwcount(dpochi, 'SessionId').reset_index()
        rwtutti = rwcount(data_tot, 'SessionId').reset_index()
        l1 = plt.scatter(rwta.SessionId, rwta.Ratio)
        l2 = plt.scatter(rwpo.SessionId, rwpo.Ratio)
        l3 = plt.scatter(rwtutti.SessionId, rwtutti.Ratio, marker='x')
        plt.legend([l1, l2, l3], ['Almeno {0} sess'.format(soglia_s),
                                  'Meno di {0} sess'.format(soglia_s),
                                  'Tutti'])
        plt.show()
# }}}
embed()
