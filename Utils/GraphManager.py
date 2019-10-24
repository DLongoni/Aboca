#!/usr/bin/env python
# coding: utf-8

# {{{ Import
from DA import DataHelper as dh
from DA import CsvLoader as CsvL
from DA import Prodotti
from DA import Users
from Utils import Constants as co

import numpy as np
import pandas as pd
import seaborn as sns  # NOQA
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt  # NOQA
from matplotlib import ticker as ticker

from IPython import embed  # NOQA
# }}}

sns.set()
sns.set_style('ticks')
sns.set_palette(co.abc_l)


def progresso(df, soglia, ax=None):
    if ax is None:
        f = plt.figure(figsize=(9, 8))
        ax = f.add_subplot(111)
    dsess_max = df.groupby('UserId')['SessionId'].max().reset_index()
    urw = dh.rwcount_base(df, 'UserId', 'Product')
    sessratio = pd.merge(dsess_max, urw, on='UserId')
    utanti = sessratio[sessratio.SessionId >= soglia].UserId.values
    upochi = sessratio[sessratio.SessionId < soglia].UserId.values
    dtanti = df[df.UserId.isin(utanti)]
    dpochi = df[df.UserId.isin(upochi)]
    rwta = dh.rwcount_base(dtanti, 'SessionId', 'Product').reset_index()
    rwpo = dh.rwcount_base(dpochi, 'SessionId', 'Product').reset_index()
    rwtutti = dh.rwcount_base(df, 'SessionId', 'Product').reset_index()
    rwtutti = rwtutti[rwtutti.SessionId < soglia]
    l1 = ax.bar(rwta.SessionId, rwta.Ratio, zorder=1)
    l2 = ax.scatter(rwpo.SessionId, rwpo.Ratio, zorder=3, s=40,
                    facecolor=co.ab_colors['rosso'], marker='v')
    l3 = ax.scatter(rwtutti.SessionId, rwtutti.Ratio, marker='o', zorder=2,
                    facecolor=co.ab_colors['giallo'], s=40)
    ax.legend([l1, l2, l3], ['Almeno {0} sessioni'.format(soglia),
                             'Meno di {0} sessioni'.format(soglia),
                             'Tutti'], fontsize=14, loc='lower right')
    ax.set_title('Correttezza media al crescere dell\'impegno', size=20)
    ax.set_ylabel('Correttezza', size=16)
    ax.set_xlabel('Sessioni', size=16)
    vals = [0, 0.25, 0.5, 0.75, 1]
    ax.set_yticks(vals)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])


def region_count(df, title="", color="", ax=None):
    if title == "":
        title = 'Numero di utenti'
    if color == "":
        color = "verde"
    if ax is None:
        f = plt.figure(figsize=(9, 8))
        ax = f.add_subplot(111)
    df = df.sort_values('Regione')
    ax.barh(df.Regione, df.Count, color=co.ab_colors[color])
    xtext = df.Count.max()/150
    for i, i_name in enumerate(df.Regione):
        i_lbl = "{0}".format(i_name)
        ax.text(xtext, i, i_lbl, color="k", va="center", size=16)

    ax.set_yticks([])
    __barh_ax_set(ax, title)


def region_corr(df, ax=None):
    title = 'Correttezza'
    if ax is None:
        f = plt.figure(figsize=(9, 8))
        ax = f.add_subplot(111)
    df = df.sort_values('Regione')
    ax.barh(df.Regione, df.Ratio*100, color=co.ab_colors['giallo'])
    for i, i_name in enumerate(df.Regione):
        i_lbl = "{0}".format(i_name)
        ax.text(1, i, i_lbl, color="k", va="center", size=16)

    __barh_ax_set(ax, title)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xticks([0, 50, 75, 100])
    xgrid = ax.xaxis.get_gridlines()
    xgrid[1].set_color('r')
    xgrid[1].set_ls('--')
    xgrid[1].set_lw(2)
    xgrid[2].set_color('r')
    xgrid[2].set_ls('--')
    xgrid[2].set_lw(2)
    ax.set_xlim([0, 100])
    ax.set_yticks([])


def prod_count(df, title="", ax=None):
    if title == "":
        title = 'Numero di prodotti'
    if ax is None:
        f = plt.figure(figsize=(9, 8))
        ax = f.add_subplot(111)
    df = df.sort_values('nTot')
    ax.barh(df.ProductId, df.nTot)
    for i, (i_name, i_tot, i_right) in enumerate(
            zip(df.ProdName, df.nTot, df.RightCount)):

        i_hist = ax.get_children()[i]
        if i_right == 0:
            i_hist.set_color(co.ab_colors['rosso'])
        else:
            i_hist.set_color(co.ab_colors['azzurro'])
        i_lbl = "{0} - {1}".format(int(i_tot), i_name)
        ax.text(1, i, i_lbl, color="k", va="center", size=16)

    __barh_ax_set(ax, title)
    ax.tick_params(labelsize=16)
    # ax.set_xticks([0, 50, 75, 100])
    ax.set_xlabel(r'Numero di prodotti', size=18)
    ax.set_yticks([])

    l_hand = []
    r_patch = mpatches.Patch(color=co.ab_colors['azzurro'], lw=1, ec='k',
                             label='Corretto')
    w_patch = mpatches.Patch(color=co.ab_colors['rosso'], lw=1, ec='k',
                             label='Errato')
    l_hand.append(r_patch)
    l_hand.append(w_patch)
    ax.legend(handles=l_hand, fontsize=16, edgecolor='k')


def freq_hist(df, title="", ylbl="", ax=None):
    if title == "":
        title = 'I prodotti più comuni sono consigliati correttamente?'
    if ylbl == "":
        ylbl = 'Numero di consigli'
    if ax is None:
        f = plt.figure(figsize=(9, 8))
        ax = f.add_subplot(111)
    df = df.sort_values('nTot')
    ax.barh(df.ProductId, df.Ratio * 100)
    for i, (i_name, i_tot) in enumerate(zip(df.ProdName, df.nTot)):

        i_lbl = "{0} - {1}".format(int(i_tot), i_name)
        ax.text(1, i, i_lbl, color="k", va="center", size=16)

    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    __barh_ax_set(ax, title)
    ax.set_xticks([0, 50, 75, 100])
    xgrid = ax.xaxis.get_gridlines()
    xgrid[1].set_color('r')
    xgrid[1].set_ls('--')
    xgrid[1].set_lw(2)
    xgrid[2].set_color('r')
    xgrid[2].set_ls('--')
    xgrid[2].set_lw(2)
    ax.set_yticks([])
    ax.set_xlim([0, 100])
    ax.set_xlabel(r'Correttezza', size=18)
    ax.set_ylabel(ylbl, size=18)


def av_freq_hist(df, title="", ylbl="", legend=True):
    if title == "":
        title = 'Gli avatar più giocati hanno ricevuto prodotti corretti?'
    if ylbl == "":
        ylbl = 'Numero di prodotti consigliati'
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(111)
    df = df.sort_values('nTot')
    ax.barh(range(0, len(df)), df.Ratio * 100)
    for i, (i_pce) in enumerate(df.AvatarPce):
        i_hist = ax.get_children()[i]
        i_hist.set_color(co.abc_l[i_pce - 1])
        i_hist.set_height(0.8)
        i_hist.set_edgecolor('k')
        i_hist.set_linewidth(1)

    if legend:
        l_hand = []
        for i in range(0, 4):
            i_patch = mpatches.Patch(color=co.abc_l[i], lw=1, ec='k',
                                     label=dh.pce_descr(i + 1))
            l_hand.append(i_patch)

        ax.legend(handles=l_hand, fontsize=16, edgecolor='k')
    for i, (i_name, i_tot, i_sess) in enumerate(
            zip(df.AvName, df.nTot, df.SessionId)):

        i_lbl = "{0} - {1} - {2}".format(int(i_tot), i_name, i_sess)
        ax.text(1, i - 0.1, i_lbl, color="k", va="center", size=14)

    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    __barh_ax_set(ax, title)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
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
    ax.set_xlim([0, 100])
    ax.set_ylim([-0.75, len(df) - 0.25])
    ax.set_xlabel(r'Correttezza', size=18)
    ax.set_ylabel(ylbl, size=18)
    f.tight_layout()


def prod_plot(uid, uhist, most_uprod):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    cmap = [0, 1, 2, 4]
    for i, (ip, ia) in enumerate(zip(most_uprod, [ax1, ax2, ax3, ax4])):
        idup = uhist[uhist.ProductId == ip]
        idup_rw = idup.groupby('SessionId').apply(lambda x: pd.Series(
            {'Ratio': sum(x.ActionType == 'RightProduct') / x.Id.count(),
             'nTot': x.Id.count()})).reset_index()
        ic = co.abc_l[cmap[i]]
        r_obs2 = range(0, len(idup_rw))
        r_obs = np.arange(-0.7, len(idup_rw) - 0.7, 1)
        ia.bar(r_obs, idup_rw.Ratio, color=ic, width=0.7, align='edge')
        ia2 = ia.twinx()
        ia2.bar(r_obs2, idup_rw.nTot, color=co.ab_colors['giallo'],
                width=0.25, align='edge')
        ia2.set_yticks(range(0, int(idup_rw.nTot.max() + 1)))
        ipname = Prodotti.get_product_name(ip)
        ia.set_title('Progresso per {0}'.format(ipname), size=22)
        ia.set_xticks([])
        ia.tick_params(labelsize=16)
        vals = [0, 0.25, 0.5, 0.75, 1]
        ia.set_yticks(vals)
        ia.yaxis.set_major_formatter(ticker.PercentFormatter())
        ia.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    a = f.axes
    a[0].set_ylabel('Correttezza', size=18)
    a[2].set_ylabel('Correttezza', size=18)
    a[5].set_ylabel('Numero prodotti consigliati', size=18,
                    color=co.ab_colors['giallo'])
    a[7].set_ylabel('Numero prodotti consigliati', size=18,
                    color=co.ab_colors['giallo'])
    a[2].set_xlabel('Sessioni', size=18)
    a[3].set_xlabel('Sessioni', size=18)

    f.suptitle('Utente {0}'.format(Users.get_user_name(uid)), size=25)


def plot_uhist(uid, i_uh, arr_start):
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
    line_avg, = ax.plot(r_obs, i_movavg, lw=3, zorder=10,
                        color=co.ab_colors['rosso'])
    bar_ratio = ax.bar(r_obs2, i_uh.Ratio, color=co.ab_colors['azzurro'],
                       alpha=0.5, width=0.7, align='edge', zorder=1)
    bar_ntot = ax2.bar(r_obs, i_uh.nTot, color=co.ab_colors['verde'],
                       alpha=0.5, width=0.25, align='edge')
    weblog = CsvL.get_web_log(uid)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    l_hand.append(line_avg)
    l_lab.append('Rolling average')
    l_hand.append(bar_ratio)
    l_lab.append('Correctness')
    l_hand.append(bar_ntot)
    l_lab.append('Number of products')
    if weblog is not None:
        num_web_check = __count_occurrences(arr_start.values, weblog)
        xval = np.add(np.where(num_web_check > 0), 0.25).squeeze(axis=0)
        yval = np.ones(len(xval)) * 0.5
        sca_web = ax.scatter(xval, yval, marker='d', edgecolors='k', s=150,
                             lw=1, facecolor=co.ab_colors['giallo'], zorder=2)
        l_hand.append(sca_web)
        l_lab.append('Website check')

    ax.legend(tuple(l_hand), tuple(l_lab), fontsize=16)
    ax.set_ylabel('% of correct recommendations', size=18)
    ax2.set_ylabel('Number of recommended products', size=18)
    ax.set_xlabel('Sessions', size=18)
    ax.set_xticks(r_obs)
    ax.tick_params(labelsize=16)
    vals = [0, 0.25, 0.5, 0.75, 1]
    ax.set_yticks(vals)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    ax2.tick_params(labelsize=16)
    yticks = np.arange(0, i_uh.nTot.max(), 5)
    yticks = np.append(yticks, i_uh.nTot.max())
    ax2.set_yticks(yticks)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    lbl = arr_start.dt.strftime('%d/%m/%Y')
    ax.set_xticklabels(lbl, rotation=45, ha='right', size=15)

    plt.title("Performance history for user"
              " {0}".format(Users.get_user_name(uid)), size=25, y=1.02)


def __count_occurrences(lb, data):
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


def __barh_ax_set(ax, title):
    ax.tick_params(labelsize=16)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(False)
    ax.set_title(title, size=20)
