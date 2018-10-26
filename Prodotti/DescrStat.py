#!/usr/bin/env python
# coding: utf-8

# {{{ Import
import seaborn as sns  # NOQA
from matplotlib import pyplot as plt  # NOQA
import pandas as pd
from DA import Prodotti
from DA import Users
from DA import DataHelper as dh
from Utils import Constants
from IPython import embed  # NOQA
import Utils.GraphManager as gm
# }}}

sns.set()
sns.set_style('ticks')
sns.set_palette(Constants.abc_l)


def rwcount(df, group, count_col='Id'):
    return dh.rwcount_base(df, group, 'Product', count_col)


df = Prodotti.get_df()  # dataframe base

# {{{ Grafico prodotti piu frequenti
prod_count = Prodotti.get_df_group_prod()
piu_frequenti = prod_count[prod_count.nTot >=
                           prod_count.quantile(0.75).nTot]

if 0:
    gm.freq_hist(piu_frequenti)
# }}}

# {{{ Prodotti più rari
if 1:
    df_rari = Prodotti.get_df_group_prod(include_rare=True).sort_values('nTot')
    p_rari = df_rari.nsmallest(10, 'nTot')[['ProdName', 'nTot']]
    print('*** Prodotti consigliati rarmente ***')
    print(p_rari)
# }}}

# {{{ Analisi temporale per utente
uhist = df[['UserId', 'AvSessId', 'SessionId', 'YMD']].drop_duplicates()
uhist_g = uhist.groupby('UserId')[['AvSessId', 'SessionId', 'YMD']].nunique()
uhist_g = uhist_g.reset_index()
uhist_g['AvPerDay'] = uhist_g.AvSessId / uhist_g.YMD
uhist_g['SessPerDay'] = uhist_g.SessionId / uhist_g.YMD
uhist_g['AvPerSess'] = uhist_g.AvSessId / uhist_g.SessionId


def plot_uhist(uid):
    uhist = Prodotti.get_user_history(uid)
    uhistrw = rwcount(uhist, 'SessionId').fillna(0)
    arr_start = dh.sess_start(uhist)
    gm.plot_uhist(uid, uhistrw, arr_start)


def prod_plot(uid):
    uhist = Prodotti.get_user_history(uid)
    uphistrw = rwcount(uhist, 'ProductId').fillna(0).reset_index()
    most_uprod = uphistrw.nlargest(4, 'nTot').ProductId.values
    gm.prod_plot(uid, uhist, most_uprod)


u_most_sess = uhist_g.nlargest(4, 'SessionId').UserId.values
if 0:
    for i_u in u_most_sess:
        plot_uhist(i_u)
# }}}

# {{{ Analisi per Avatar
av = rwcount(df, 'AvSessId')
av_top = av[av.nTot >= av.nTot.quantile(.80)].reset_index()
av_top_pce = dh.add_avatar_data(av_top)
av_worst = int(av_top[av_top.Ratio == av_top.Ratio.min()].AvSessId)
df_worst_rw = dh.top_prod_av_breakdown(df, av_worst)
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

# analisi per pce, per prodotto
if 1:
    for i_pce in all_pce:
        data_rwt = df[df.AvatarPce == i_pce]
        apce = rwcount(data_rwt, 'ProductId')
        apce = apce[apce.nTot > apce.quantile(0.8).nTot].reset_index()
        apce = dh.add_prod_name(apce)

        i_tit = "I prodotti più consigliati per il PCE {0}".format(
            dh.pce_descr(i_pce))
        gm.freq_hist(apce, i_tit)
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


user_ratio = rwcount(df, 'UserId').reset_index()
u_rw_hist = pd.merge(user_ratio, uhist_g)
udata = Users.get_users_table()
udmerge = pd.merge(u_rw_hist, udata)
regionymd = udmerge.groupby('Regione').YMD.mean()
provinceymd = udmerge.groupby('ProvId').YMD.mean()
roleymd = udmerge.groupby('RoleId').YMD.mean()
# }}}
embed()
