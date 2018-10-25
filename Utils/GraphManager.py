#!/usr/bin/env python
# coding: utf-8

# {{{ Import
import seaborn as sns  # NOQA
from matplotlib import pyplot as plt  # NOQA
from matplotlib import ticker as ticker
from IPython import embed  # NOQA
# }}}

sns.set()
sns.set_style('ticks')


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
