#!/usr/bin/env python

import pandas as pd
from IPython import embed # NOQA


def get_users_table():
    # utenti
    u = pd.read_csv('./Dataset/Dumps/out_qAbpUsers.csv', sep='$')
    # farmacie
    f = pd.read_csv('./Dataset/Dumps/out_qVrCustomers.csv', sep='$')
    # province
    p = pd.read_csv('./Dataset/Dumps/out_qProvince.csv', sep='$')
    # regioni
    r = pd.read_csv('./Dataset/Dumps/out_qRegioni.csv', sep='$')
    # userroles
    ur = pd.read_csv('./Dataset/Dumps/out_qAbpUserroles.csv', sep='$')

    u = u[['Id', 'ClientCode', 'PdcCode']]
    u.rename(columns={'Id': 'UserId'}, inplace=True)
    u = u[~u.ClientCode.isnull()]
    u['UserId'] = u.UserId.astype(int)
    f = f[['Id', 'Code', 'PdcCode', 'Province']]
    f.rename(columns={'Id': 'FarmaId'}, inplace=True)
    f.rename(columns={'Province': 'ProvId'}, inplace=True)
    p.drop(['Id', 'Nome', 'Longitudine'], axis=1, inplace=True)
    p = p[['Id_Regione', 'Sigla_automobilistica', 'Latitudine']]
    p.rename(columns={'Sigla_automobilistica': 'ProvId'}, inplace=True)
    r['Nome'] = r.Nome.apply(lambda n: n[0:4])
    ur.drop(['Id', 'CreatorUserId', 'TenantId'], axis=1, inplace=True)

    # alcuni utenti tipo 208 hanno più di un ruolo
    ur = ur.groupby('UserId')['RoleId'].last().reset_index()

    uf = pd.merge(u, f, left_on=['ClientCode', 'PdcCode'],
                  right_on=['Code', 'PdcCode'], how='left')
    ufp = pd.merge(uf, p)
    ufpr = pd.merge(ufp, r)
    ufprr = pd.merge(ufpr, ur)
    ufprr.drop(['Id_Regione'], axis=1, inplace=True)
    ufprr.rename(columns={'Nome': 'Regione'}, inplace=True)
    ufprr.drop(['Code', 'PdcCode'], axis=1, inplace=True)
    return ufprr


def merge_users_clean(df):
    udf = get_users_table()
    df = pd.merge(df, udf, left_on='UserId', right_on='UserId')
    df = df[df.RoleId.isin([7, 8, 9])]
    return df


def clean_df_roles(df):
    df = df[df.RoleId.isin([7, 8, 9])]
    return df


def clean_df_userid(df):
    udf = get_users_table()
    mapping = dict(udf[['UserId', 'RoleId']].values)
    df['RoleId'] = df.UserId.replace(mapping)
    df = df[df.RoleId.isin([7, 8, 9])]
    df = df.drop('RoleId', axis=1)
    return df


if __name__ == "__main__":
    d = get_users_table()
