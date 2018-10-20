#!/usr/bin/env python

# {{{ Import
import numpy as np
from sklearn.preprocessing import scale

import pandas as pd
from DA import Users
from DA import Avatar
from DA import DatesManager as dm
# }}}


def get_df(max_date=-1):
    df = pd.read_csv('./Dataset/Dumps/out_qVrAvatarProduct.csv', sep='$')
    df = Avatar.merge_avatar(df, cut_pce=[6, 7])
    df.drop(['TenantId', 'DeletionTime', 'LastModificationTime', 'SessionId',
             'AvatarId', 'LastModifierUserId', 'CreatorUserId', 'IsDeleted',
             'DeleterUserId', 'ProductSequence', 'ProductPce'],
            axis=1, inplace=True)
    df = df.drop(df.index[df.ProductType == 'RecommendedProduct'])
    df = df.drop(df.index[df.ProductType == 'SoldProduct'])
    df.CreationTime = pd.to_datetime(df.CreationTime, dayfirst=True)
    df = dm.filter_date(df, 'CreationTime', max_date)
    df = dm.add_aggregate_date(df, 'CreationTime')
    df = Users.merge_users_clean(df)
    return df


def get_df_group_prod(df=None, include_rare=False):
    if df is None:
        df = get_df()

    p_anag = pd.read_csv('./Dataset/Dumps/ProdAnag.csv', sep='$')
    p_anag['ProductId'] = p_anag['Codice'] + p_anag['Azienda']
    p_anag['ProdName'] = (p_anag['Descrizione'] + ' ' + p_anag['Formato'] +
                          ' ' + p_anag['Confezione'])
    p_anag = p_anag[['ProductId', 'ProdName', 'Prezzo']]
    p_anag = p_anag.groupby('ProductId').first().reset_index()

    # Names
    # prod = df[['ProductId', 'ProductName', 'ProductFormat']]
    # prod = prod.groupby('ProductId')['ProductName',
    #                                  'ProductFormat'].first().reset_index()
    # prod['Name'] = prod['ProductName'] + ' ' + prod['ProductFormat']
    # prod.drop(['ProductName', 'ProductFormat'], axis=1, inplace=True)
    # prod.set_index('ProductId',true)

    n_users = df.groupby('ProductId')['UserId'].nunique().reset_index()
    n_users.rename(columns={'UserId': 'nUsers'}, inplace=True)

    n_farma = df.groupby('ProductId')['FarmaId'].nunique().reset_index()
    n_farma.rename(columns={'FarmaId': 'nFarma'}, inplace=True)

    n_tot = df.groupby('ProductId')['Id'].count().reset_index()
    n_tot.rename(columns={'Id': 'nTot'}, inplace=True)

    df_r = df[df.ProductType == 'RightProduct']
    n_r = df_r.groupby('ProductId')['Id'].count().reset_index()
    n_r.rename(columns={'Id': 'nRight'}, inplace=True)

    lat_m = df.groupby('ProductId')['Latitudine'].mean().reset_index()
    # Ipotetica latitudine di centro italia
    lat_m.Latitudine = lat_m.Latitudine - 42
    lat_m.rename(columns={'Latitudine': 'NordSud'}, inplace=True)

    lat_v = df.groupby('ProductId')['Latitudine'].var().reset_index()
    lat_v = lat_v.fillna(0)
    lat_v.rename(columns={'Latitudine': 'LatVar'}, inplace=True)

    n_prov = df.groupby('ProductId')['ProvId'].nunique().reset_index()
    n_prov.rename(columns={'ProvId': 'nProv'}, inplace=True)

    n_reg = df.groupby('ProductId')['Regione'].nunique().reset_index()
    n_reg.rename(columns={'Regione': 'nReg'}, inplace=True)

    prod = pd.merge(p_anag, n_users)
    prod = pd.merge(prod, n_farma)
    prod = pd.merge(prod, n_tot)
    prod = pd.merge(prod, n_r)
    prod = pd.merge(prod, lat_m)
    prod = pd.merge(prod, lat_v)
    prod = pd.merge(prod, n_prov)
    prod = pd.merge(prod, n_reg)
    prod['Ratio'] = prod.nRight/prod.nTot
    prod['UserRatio'] = prod.nTot/prod.nUsers
    prod['GeoRatio'] = prod.nTot/prod.nProv
    # pd.plotting.scatter_matrix(prod,diagonal = 'kde')
    # pyplot.show()
    # nFarma e nUsers sono sostituibili
    # seleziono quelli su cui ha senso fare un'analisi. corretto o no?
    if not include_rare:
        prod = prod[prod.nTot > 2].reset_index(drop=True)
    return prod


def get_df_group_prod_proc(prod=None):
    if prod is None:
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
    # pd.plotting.scatter_matrix(prod_proc,diagonal = 'kde')
    # pyplot.show()
    return prod_proc
