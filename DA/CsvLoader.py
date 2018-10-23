#!/usr/bin/env python

from functools import lru_cache
from IPython import embed
import pandas as pd


# {{{ REGION: Csv get and cache

@lru_cache(maxsize=100)
def get_prod_anag():
    print('*** Loading products data from csv')
    p_anag = pd.read_csv('./Dataset/Dumps/ProdAnag.csv', sep='$')
    p_anag['ProductId'] = p_anag['Codice'] + p_anag['Azienda']
    p_anag['ProdName'] = (p_anag['Descrizione'] + ' ' + p_anag['Formato'] +
                          ' ' + p_anag['Confezione'])
    p_anag = p_anag[['ProductId', 'ProdName']]
    p_anag = p_anag.groupby('ProductId').first().reset_index()
    return p_anag


@lru_cache(maxsize=100)
def get_prod_history():
    print('*** Loading products history from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qVrAvatarProduct.csv', sep='$')
    df.drop(['TenantId', 'DeletionTime', 'LastModificationTime',
             'LastModifierUserId', 'CreatorUserId', 'IsDeleted',
             'DeleterUserId', 'ProductSequence', 'ProductPce'],
            axis=1, inplace=True)
    df = df.drop(df.index[df.ProductType == 'RecommendedProduct'])
    df = df.drop(df.index[df.ProductType == 'SoldProduct'])
    df.CreationTime = pd.to_datetime(df.CreationTime, dayfirst=True)
    hard_fix_prod_hist(df)
    return df


@lru_cache(maxsize=100)
def get_avatar_pce():
    print('*** Loading avatar pce from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qVrAvatarsPce.csv', sep='$')
    df = df[['Id', 'SessionId', 'AvatarId', 'PCEId']]
    df.rename(columns={'Id': 'AvSessId'}, inplace=True)
    df.rename(columns={'PCEId': 'AvatarPce'}, inplace=True)
    return df


@lru_cache(maxsize=100)
def get_avatar_anag():
    print('*** Loading avatar data from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qVrAvatars.csv', sep='$')
    df = df[['Id', 'Name', 'Surname', 'Age']]
    df['AvName'] = df['Name'] + ' ' + df['Surname']
    df.drop(['Name', 'Surname'], axis=1, inplace=True)
    df.rename(columns={'Id': 'AvatarId'}, inplace=True)
    return df


@lru_cache(maxsize=100)
def get_users_anag():
    print('*** Loading users anag from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qAbpUsers.csv', sep='$')
    return df


@lru_cache(maxsize=100)
def get_customers_anag():
    print('*** Loading customers anag from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qVrCustomers.csv', sep='$')
    return df


@lru_cache(maxsize=100)
def get_province():
    print('*** Loading province from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qProvince.csv', sep='$')
    return df


@lru_cache(maxsize=100)
def get_regions():
    print('*** Loading regions from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qRegioni.csv', sep='$')
    return df


@lru_cache(maxsize=100)
def get_user_roles():
    print('*** Loading user roles from csv')
    df = pd.read_csv('./Dataset/Dumps/out_qAbpUserroles.csv', sep='$')
    return df

# }}}


# {{{ REGION: Data Hardfix
# Purtroppo tocca correggere i dati a mano. Lo faccio a monte di tutto.
def hard_fix_prod_hist(df):
    df.loc[(df.ProductId == 'P0026AB') & (df.AvatarId == 18) &
           (df.SessionId == 4) & (df.ProductType == 'WrongProduct'),
           'ProductType'] = 'RightProduct'

# }}}


if __name__ == '__main__':
    embed()