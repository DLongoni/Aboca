#!/usr/bin/env python
# coding: utf-8

# {{{ Import
import seaborn as sns  # NOQA
from matplotlib import pyplot as plt  # NOQA
import pandas as pd

from DA import Info
from DA import DataHelper as dh
from Utils import Constants
import Utils.GraphManager as gm
from IPython import embed  # NOQA
# }}}

sns.set()
sns.set_style('ticks')
sns.set_palette(Constants.abc_l)


def rwcount(df, group, type_suffix='Info', count_col='Id'):
    return dh.rwcount_base(df, group, type_suffix, count_col)


df = Info.get_df()  # dataframe base


# {{{ Grafico prodotti piu frequenti
if 0:
    rw_info = rwcount(df, 'ProductId', "Info")
    rw_info_freq = dh.add_prod_name(rw_info.nlargest(20, 'nTot'))
    gm.freq_hist(rw_info_freq, "Le informazioni sui prodotti più frequenti "
                 "sono corrette?", "Numero di informazioni date")

if 0:
    rw_info = rwcount(df, 'ProductId', "Benefit")
    rw_info_freq = dh.add_prod_name(rw_info.nlargest(20, 'nTot'))
    gm.freq_hist(rw_info_freq, "I benefici indicati per i prodotti "
                 "più frequenti sono corretti?", "Numero di benefici indicati")

if 0:
    rw_info = rwcount(df, 'ProductId', "Advantages")
    rw_info_freq = dh.add_prod_name(rw_info.nlargest(20, 'nTot'))
    gm.freq_hist(rw_info_freq, "I vantaggi indicati per i prodotti "
                 "più frequenti sono corretti?", "Numero di vantaggi indicati")
# }}}

# {{{ Analisi temporale per prodotto e utente
uhist = df[['UserId', 'AvSessId', 'SessionId', 'YMD']].drop_duplicates()
uhist_g = uhist.groupby('UserId')[['AvSessId', 'SessionId', 'YMD']].nunique()
uhist_g = uhist_g.reset_index()
uhist_g['AvPerDay'] = uhist_g.AvSessId / uhist_g.YMD
uhist_g['SessPerDay'] = uhist_g.SessionId / uhist_g.YMD
uhist_g['AvPerSess'] = uhist_g.AvSessId / uhist_g.SessionId


# TODO: titoli e descrizioni varie
def plot_uhist(uid, type_suffix):
    uhist = Info.get_user_history(uid)
    uhistrw = rwcount(uhist, 'SessionId', type_suffix).fillna(0)
    arr_start = dh.sess_start(uhist)
    gm.plot_uhist(uid, uhistrw, arr_start)


def prod_plot(uid, type_suffix):
    uhist = Info.get_user_history(uid)
    uphistrw = rwcount(uhist, 'ProductId', type_suffix).fillna(0).reset_index()
    most_uprod = uphistrw.nlargest(4, 'nTot').ProductId.values
    gm.prod_plot(uid, uhist, most_uprod)


u_most_sess = uhist_g.nlargest(4, 'SessionId').UserId.values
if 0:
    for i_u in u_most_sess:
        plot_uhist(i_u)
# }}}

# {{{ Analisi per Avatar
av = rwcount(df, 'AvSessId', 'Info')
av_top = av.nlargest(20, 'nTot').reset_index()
av_top_pce = dh.add_avatar_data(av_top)
av_worst = int(av_top[av_top.Ratio == av_top.Ratio.min()].AvSessId)
df_worst_rw = dh.top_prod_av_breakdown(df, av_worst, 10, "Info")
print("*** Prodotti più consigliati erroneamente all'avatar peggiore "
      "[{0}] ***".format(str(av_worst)))
print(df_worst_rw)
# }}}

# {{{ Analisi per PCE
all_pce = sorted(df.AvatarPce.unique())

# analisi per pce, per avatar - poco interessante??
if 0:
    for i_pce in all_pce:
        data_rwt = df[df.AvatarPce == i_pce]
        apce = rwcount(data_rwt, 'AvSessId')
        apce = apce[apce.nTot > apce.quantile(0.8).nTot].reset_index()
        apce = dh.add_avatar_data(apce)
        i_tit = "Avatar più giocati per il PCE {0}".format(
            dh.pce_descr(i_pce))
        gm.av_freq_hist(apce, i_tit, False)

# analisi per pce, per prodotto - c'è da fidarsi??
if 0:
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    for i_pce, iax in enumerate([ax1, ax2, ax3, ax4]):
        data_rwt = df[df.AvatarPce == i_pce+1]
        apce = rwcount(data_rwt, 'ProductId')
        apce = apce.nlargest(8, 'nTot').reset_index()
        apce = dh.add_prod_name(apce)

        i_tit = "PCE {0}".format(
            dh.pce_descr(i_pce+1))
        gm.freq_hist(apce, i_tit, iax)
        for i in range(0, 8):
            i_hist = iax.get_children()[i]
            i_hist.set_color(Constants.abc_l[i_pce])

    f.suptitle("I prodotti più consigliati per PCE", size=22)
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax4.set_ylabel("")
# }}}

# {{{ Analisi geografica
# TODO: grafico bar o barh prov con colormap ratio e ntot lunghezza barre
prov = rwcount(df, 'Regione')

users_per_reg = df.groupby('Regione')['UserId'].nunique().reset_index()
users_per_reg.set_index('Regione', inplace=True)

prov = pd.merge(prov, users_per_reg, left_index=True, right_index=True)
rol = rwcount(df, 'RoleId')
# }}}

# {{{ Analisi num giocate utenti
ymdcount = uhist_g.YMD.value_counts()
avcount = uhist_g.AvSessId.value_counts()
# Il grafico della morte
# plt.scatter(ymdcount.index, ymdcount.values)


# OSS: mediare sulle sessioni direttametne e mediare su [sessioni, utenti]
# per poi mediare sulle sessioni porta a risultati molto simili. Il primo
# è un progresso mediato sulle giocate e il secondo sul giocatore, circa.
# comunque non cambia quasi niente
if 0:
    for soglia_s in [11, 19]:
        dsess_max = df.groupby('UserId')['SessionId'].max().reset_index()
        urw = rwcount(df, 'UserId')
        sessratio = pd.merge(dsess_max, urw, on='UserId')
        utanti = sessratio[sessratio.SessionId >= soglia_s].UserId.values
        upochi = sessratio[sessratio.SessionId < soglia_s].UserId.values
        dtanti = df[df.UserId.isin(utanti)]
        dpochi = df[df.UserId.isin(upochi)]
        rwta = rwcount(dtanti, 'SessionId').reset_index()
        rwpo = rwcount(dpochi, 'SessionId').reset_index()
        rwtutti = rwcount(df, 'SessionId').reset_index()
        l1 = plt.scatter(rwta.SessionId, rwta.Ratio)
        l2 = plt.scatter(rwpo.SessionId, rwpo.Ratio)
        l3 = plt.scatter(rwtutti.SessionId, rwtutti.Ratio, marker='x')
        plt.legend([l1, l2, l3], ['Almeno {0} sess'.format(soglia_s),
                                  'Meno di {0} sess'.format(soglia_s),
                                  'Tutti'])
        plt.show()

# udata = Users.get_users_table()
# udmerge = pd.merge(u_rw_hist, udata)
# regionymd = udmerge.groupby('Regione').YMD.mean()
# provinceymd = udmerge.groupby('ProvId').YMD.mean()
# roleymd = udmerge.groupby('RoleId').YMD.mean()
# }}}
embed()
