#!/usr/bin/env python
# coding: utf-8

# {{{ Import
from DA import DataHelper as dh
from DA import CsvLoader as CsvL
from DA import Prodotti
from DA import Users
from Utils import Constants

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
sns.set_palette(Constants.abc_l)


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
        i_hist.set_color(Constants.abc_l[i_pce-1])
        # i_hist.set_alpha(0.7)
        i_hist.set_height(0.8)
        i_hist.set_edgecolor('k')
        i_hist.set_linewidth(1)

    if legend:
        l_hand = []
        for i in range(0, 4):
            i_patch = mpatches.Patch(color=Constants.abc_l[i], lw=1, ec='k',
                                     label=dh.pce_descr(i + 1))
            l_hand.append(i_patch)

        ax.legend(handles=l_hand, fontsize=14, edgecolor='k')
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


def prod_plot(uid, uhist, most_uprod):
    plt.ion()
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    for i, (ip, ia) in enumerate(zip(most_uprod, [ax1, ax2, ax3, ax4])):
        idup = uhist[uhist.ProductId == ip]
        idup_rw = idup.groupby('SessionId').apply(lambda x: pd.Series(
            {'Ratio': sum(x.ActionType == 'RightProduct') / x.Id.count(),
             'nTot': x.Id.count()})).reset_index()
        ic = Constants.abc_l[i]
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


def plot_uhist(uid, i_uh, arr_start):
    plt.ion()
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
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    l_hand.append(line_avg)
    l_lab.append('Rolling average')
    l_hand.append(bar_ratio)
    l_lab.append('Correttezza')
    l_hand.append(bar_ntot)
    l_lab.append('Tot prodotti')
    if weblog is not None:
        num_web_check = __count_occurrences(arr_start.values, weblog)
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
