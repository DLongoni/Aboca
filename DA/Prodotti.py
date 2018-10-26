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
             'RoleId', 'FarmaId', 'Latitudine']]
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

    n_users = df.groupby('ProductId')['UserId'].nunique().reset_index()
    n_users.rename(columns={'UserId': 'nUsers'}, inplace=True)

    n_farma = df.groupby('ProductId')['FarmaId'].nunique().reset_index()
    n_farma.rename(columns={'FarmaId': 'nFarma'}, inplace=True)

    n_tot = df.groupby('ProductId')['Id'].count().reset_index()
    n_tot.rename(columns={'Id': 'nTot'}, inplace=True)

    df_r = df[df.ActionType == 'RightProduct']
    n_r = df_r.groupby('ProductId')['Id'].count().reset_index()
    n_r.rename(columns={'Id': 'nRight'}, inplace=True)

    lat_m = df.groupby('ProductId')['Latitudine'].mean().reset_index()
    # Ipotetica latitudine di centro italia
    lat_m.Latitudine = lat_m.Latitudine - 42
    lat_m.rename(columns={'Latitudine': 'NordSud'}, inplace=True)

    lat_v = df.groupby('ProductId')['Latitudine'].var().reset_index()
    lat_v = lat_v.fillna(0)
    lat_v.rename(columns={'Latitudine': 'LatVar'}, inplace=True)

    # n_prov = df.groupby('ProductId')['ProvId'].nunique().reset_index()
    # n_prov.rename(columns={'ProvId': 'nProv'}, inplace=True)

    n_reg = df.groupby('ProductId')['Regione'].nunique().reset_index()
    n_reg.rename(columns={'Regione': 'nReg'}, inplace=True)

    prod = pd.merge(p_anag, n_users)
    prod = pd.merge(prod, n_farma)
    prod = pd.merge(prod, n_tot)
    prod = pd.merge(prod, n_r)
    prod = pd.merge(prod, lat_m)
    prod = pd.merge(prod, lat_v)
    # prod = pd.merge(prod, n_prov)
    prod = pd.merge(prod, n_reg)
    prod['Ratio'] = prod.nRight/prod.nTot
    prod['UserRatio'] = prod.nTot/prod.nUsers
    # prod['GeoRatio'] = prod.nTot/prod.nProv
    if not include_rare:
        prod = prod[prod.nTot > 2].reset_index(drop=True)
    return prod


@lru_cache(maxsize=100)
def get_df_group_prod_proc():
    prod = get_df_group_prod()
    prod_proc = prod.copy(deep=True)
    prod_proc.nUsers = scale(np.log(prod.nUsers))
    prod_proc.nFarma = scale(np.log(prod.nFarma))
    prod_proc.nTot = scale(np.log(prod.nTot))
    prod_proc.nRight = scale(np.log(prod.nRight))
    prod_proc.NordSud = scale(prod.NordSud)
    prod_proc.LatVar = scale(prod.LatVar)
    prod_proc.nProv = scale(np.log(prod.nProv))
    prod_proc.nReg = scale(np.log(prod.nReg))
    prod_proc.Ratio = scale(prod.Ratio)
    prod_proc.UserRatio = scale(np.log(prod.UserRatio))
    prod_proc.GeoRatio = scale(np.log(prod.GeoRatio))
    return prod_proc


def get_product_name(product_id):
    p = CsvL.get_prod_anag()
    ret = p[p.ProductId == product_id].ProdName.values[0]
    return ret


if __name__ == '__main__':
    embed()
