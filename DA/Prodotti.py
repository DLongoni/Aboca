#!/usr/bin/env python

# {{{ Import
import numpy as np
from functools import lru_cache
from sklearn.preprocessing import scale

import pandas as pd
from IPython import embed
from DA import Users
from DA import DatesManager as dm
from DA import CsvLoader as CsvL
from DA import DataHelper as dh
# }}}


@lru_cache(maxsize=100)
def get_df(max_date=-1):
    df = CsvL.get_prod_history()
    df = dh.add_avatar_data(df, cut_pce=[5, 6, 7])
    __hist_hardfix(df)
    df = dh.add_session_date(df)
    df.drop(['AvatarId'], axis=1, inplace=True)
    df = dm.filter_date(df, 'YMD', max_date)
    df = Users.merge_users_clean(df)
    df = df[['Id', 'UserId', 'SessionId', 'ActionType', 'ProductId',
             'AvSessId', 'AvatarPce', 'YMD', 'NameSurname', 'Regione',
             'RoleId', 'FarmaId', 'Latitudine', 'ProvId', 'Sex']]
    df = dh.add_prod_name(df)
    return df


def get_user_history(user_id):
    df = get_df()
    user_hist = df[df.UserId == user_id]
    return user_hist


def __hist_hardfix(df):
    df.loc[(df.ProductName.str.contains('Colilen')) & (df.AvatarPce == 5),
           'ActionType'] = 'RightProduct'
    avprod_rw = df.groupby(['AvSessId', 'ProductId']).apply(
        lambda x: pd.Series(
            {'Ratio': sum(x.ActionType == 'RightProduct') / x.Id.count(),
             'nTot': x.Id.count()})).reset_index()
    avp_wrong = avprod_rw[~avprod_rw.Ratio.isin([0, 1])]
    for ia, ip in avp_wrong[['AvSessId', 'ProductId']].itertuples(index=False):
        df.loc[(df.AvSessId == ia) & (df.ProductId == ip),
               'ActionType'] = 'RightProduct'


@lru_cache(maxsize=100)
def get_df_group_prod(include_rare=False):
    df = get_df()
    p_anag = CsvL.get_prod_anag()

    feat_df = df.groupby('ProductId').apply(lambda x: pd.Series({
        'nUsers': x.UserId.nunique(),
        'nFarma': x.FarmaId.nunique(),
        'nProv': x.ProvId.nunique(),
        'nReg': x.Regione.nunique(),
        'nAvSess': x.AvSessId.nunique(),
        'nSess': x.SessionId.nunique(),
        'nTot': x.Id.count(),
        'MedianPce': x.AvatarPce.median(),
        'MeanPce': x.AvatarPce.mean(),
        'nRight': sum(x.ActionType == "RightProduct"),
        'NordSud': x.Latitudine.mean()-42,
        # 'LatVar': x.Latitudine.var(),
        'UserRatio': x.Id.count() / x.UserId.nunique(),
        'Ratio': sum(x.ActionType == 'RightProduct') / x.Id.count(),
        'Recency': (dm.MAXDATE - x.YMD.max()).days+1,
        'Frequency': x.YMD.nunique()
    })).reset_index()

    prod = pd.merge(p_anag, feat_df)
    if not include_rare:
        prod = prod[prod.nTot > 2].reset_index(drop=True)
    return prod


@lru_cache(maxsize=100)
def get_df_group_prod_proc(include_rare=False):
    prod = get_df_group_prod(include_rare)
    prod_proc = prod.copy(deep=True)
    prod_proc.nUsers = scale(np.log(prod.nUsers.values))
    prod_proc.nFarma = scale(np.log(prod.nFarma.values))
    prod_proc.nProv = scale(np.log(prod.nProv.values))
    prod_proc.nReg = scale(np.log(prod.nReg.values))
    prod_proc.nAvSess = scale(np.log(prod.nAvSess.values))
    prod_proc.nSess = scale(np.log(prod.nSess.values))
    prod_proc.nTot = scale(np.log(prod.nTot.values))
    prod_proc.MedianPce = scale(prod.MedianPce.values)
    prod_proc.MeanPce = scale(prod.MeanPce.values)
    prod_proc.nRight = scale(prod.nRight.values)
    prod_proc.NordSud = scale(prod.NordSud.values)
    # prod_proc.LatVar = scale(prod.LatVar.values)
    prod_proc.UserRatio = scale(np.log(prod.UserRatio.values))
    prod_proc.Ratio = scale(prod.Ratio.values)
    prod_proc.Recency = scale(np.log(prod.Recency.values))
    prod_proc.Frequency = scale(np.log(prod.Frequency.values))
    return prod_proc


def get_product_name(product_id):
    p = CsvL.get_prod_anag()
    ret = p[p.ProductId == product_id].ProdName.values[0]
    return ret


if __name__ == '__main__':
    embed()
