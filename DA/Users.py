#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from DA import CsvLoader as CsvL
from IPython import embed # NOQA


def get_users_table():
    u  = CsvL.get_users_anag()
    f  = CsvL.get_customers_anag()
    p  = CsvL.get_province()
    r  = CsvL.get_regions()
    ur = CsvL.get_user_roles()

    f = f[['Id', 'Code', 'PdcCode', 'Province']]
    f = f.rename(columns={'Id': 'FarmaId'})
    f = f.rename(columns={'Province': 'ProvId'})
    p = p.drop(['Id', 'Nome', 'Longitudine'], axis=1)
    p = p[['Id_Regione', 'Sigla_automobilistica', 'Latitudine']]
    p = p.rename(columns={'Sigla_automobilistica': 'ProvId'})
    # r['Nome'] = r.Nome.apply(lambda n: n[0:4])
    ur = ur.drop(['Id', 'CreatorUserId', 'TenantId'], axis=1)

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
    ufprr = ufprr[ufprr.UserId != 12]  # Andrea Dini
    return ufprr


def get_user_name(user_id):
    u = CsvL.get_users_anag()
    ret = u[u.UserId == user_id].NameSurname.values[0]
    return ret


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


def add_user_data(df):
    u = get_users_table()
    u = u[['UserId', 'NameSurname', 'Regione', 'RoleId']]
    df = pd.merge(df, u, on='UserId')
    return df


if __name__ == "__main__":
    d = get_users_table()
