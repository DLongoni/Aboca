#!/usr/bin/env python
# coding: utf-8

# {{{ Import
import seaborn as sns  # NOQA
from matplotlib import pyplot as plt  # NOQA
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
if 0:
    df_rari = Prodotti.get_df_group_prod(include_rare=True).sort_values('nTot')
    p_rari = df_rari.nsmallest(10, 'nTot')[['ProdName', 'nTot']]
    print('*** Prodotti consigliati raramente ***')
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
av_top = av.nlargest(20, 'nTot').reset_index()
av_top_pce = dh.add_avatar_data(av_top)
av_worst = int(av_top[av_top.Ratio == av_top.Ratio.min()].AvSessId)
df_worst_rw = dh.top_prod_av_breakdown(df, av_worst)
print("*** Prodotti più consigliati erroneamente all'avatar peggiore "
      "[{0}] ***".format(str(av_worst)))
print(df_worst_rw)
# }}}

# {{{ Analisi per PCE
all_pce = sorted(df.AvatarPce.unique())

# Grafico avatar più giocati per pce
if 0:
    gm.av_freq_hist(av_top_pce)

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
if 1:
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    for i_pce, iax in enumerate([ax1, ax2, ax3, ax4]):
        data_rwt = df[df.AvatarPce == i_pce+1]
        apce = rwcount(data_rwt, 'ProductId')
        apce = apce.nlargest(8, 'nTot').reset_index()
        apce = dh.add_prod_name(apce)

        i_tit = "PCE {0}".format(
            dh.pce_descr(i_pce+1))
        gm.freq_hist(apce, i_tit, ax=iax)
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
reg = rwcount(df, 'Regione')
reg = reg.sort_values('Regione')
users_per_reg = df.groupby('Regione')['UserId'].nunique().reset_index()
ureg = users_per_reg.rename(columns={'UserId': 'Count'})
udata = Users.add_user_data(uhist_g)
regav = udata.groupby('Regione').AvSessId.mean().reset_index()
regav = regav.rename(columns={'AvSessId': 'Count'})
if 0:
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    gm.region_count(ureg, ax=ax1, color='azzurro')
    gm.region_count(regav, "Numero medio di avatar giocati", ax=ax2)
    gm.region_corr(reg, ax=ax3)
    f.suptitle("Overview regionale", size=22)
    ax1.set_xlabel("Numero di utenti", size=18)
    ax2.set_xlabel("Numero di avatar", size=18)
    ax3.set_xlabel("Correttezza", size=18)
    ax1.set_title("Quanti utenti hanno giocato?", size=20)
    ax2.set_title("Clienti serviti in media", size=20)
    ax3.set_title("Correttezza dei consigli", size=20)

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
    for soglia_s in [11]:  # buono anche 19
        gm.progresso(df, soglia_s)
# }}}
embed()
