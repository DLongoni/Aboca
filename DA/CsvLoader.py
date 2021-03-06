#!/usr/bin/env python

from functools import lru_cache
from IPython import embed
from DA import DatesManager as dm
from Utils import Constants as co
import logging
import pandas as pd


# {{{ REGION: Products
@lru_cache(maxsize=100)
def get_prod_anag():
    print('*** Loading products data from csv')
    p_anag = pd.read_csv(co.dumps_path()+'ProdAnag.csv', sep='$')
    p_anag['ProductId'] = p_anag['Codice'] + p_anag['Azienda']
    p_anag['ProdName'] = (p_anag['Descrizione'] + ' ' + p_anag['Formato'] +
                          ' ' + p_anag['Confezione'])
    p_anag = p_anag[['ProductId', 'ProdName']]
    p_anag = p_anag.groupby('ProductId').first().reset_index()
    return p_anag


@lru_cache(maxsize=100)
def get_prod_history():
    print('*** Loading products history from csv')
    df = pd.read_csv(co.dumps_path()+'out_qVrAvatarProduct.csv', sep='$')
    df.drop(['TenantId', 'DeletionTime', 'LastModificationTime',
             'LastModifierUserId', 'CreatorUserId', 'IsDeleted',
             'DeleterUserId', 'ProductSequence', 'ProductPce', 'CreationTime'],
            axis=1, inplace=True)
    df = df.drop(df.index[df.ProductType == 'RecommendedProduct'])
    df = df.drop(df.index[df.ProductType == 'SoldProduct'])
    hard_fix_prod_hist(df)
    df = df.rename(columns={'ProductType': 'ActionType'})
    return df
# }}}


# {{{ REGION: Info
@lru_cache(maxsize=100)
def get_avatar_info():
    print('*** Loading avatar info from csv')
    df = pd.read_csv(co.dumps_path()+'out_qVrAvatarInfo.csv', sep='$')
    # 25-10-18 ad oggi escludo sequence, e infoid che non so come interpretare
    df = df[['Id', 'UserId', 'SessionId', 'AvatarId', 'InfoType',
             'InfoText', 'ProductId']]
    df = df[(df.InfoType.str.contains('wrong', False)) |
            (df.InfoType.str.contains('right', False))]
    df.loc[df.InfoType.str.contains(
        r'\bProduct\dRightInfo\b'), 'InfoType'] = 'RightInfo'
    df.loc[df.InfoType.str.contains(
        r'\bProduct\dWrongInfo\b'), 'InfoType'] = 'WrongInfo'
    df.loc[df.InfoType.str.contains(
        r'\bProduct\dRightAdvantages\b'), 'InfoType'] = 'RightAdvantages'
    df.loc[df.InfoType.str.contains(
        r'\bProduct\dWrongAdvantages\b'), 'InfoType'] = 'WrongAdvantages'
    df.loc[df.InfoType.str.contains(
        r'\bProduct\dRightBenefit\b'), 'InfoType'] = 'RightBenefit'
    df.loc[df.InfoType.str.contains(
        r'\bProduct\dWrongBenefit\b'), 'InfoType'] = 'WrongBenefit'
    df = df.rename(columns={'InfoType': 'ActionType'})
    return df
# }}}


# {{{ REGION: Avatar
@lru_cache(maxsize=100)
def get_avatar_pce():
    print('*** Loading avatar pce from csv')
    df = pd.read_csv(co.dumps_path()+'out_qVrAvatarsPce.csv', sep='$')
    df = df[['Id', 'SessionId', 'AvatarId', 'PCEId']]
    df.rename(columns={'Id': 'AvSessId'}, inplace=True)
    df.rename(columns={'PCEId': 'AvatarPce'}, inplace=True)
    return df


@lru_cache(maxsize=100)
def get_avatar_anag():
    print('*** Loading avatar data from csv')
    df = pd.read_csv(co.dumps_path()+'out_qVrAvatars.csv', sep='$')
    df = df[['Id', 'Name', 'Surname', 'Age', 'Sex']]
    df['AvName'] = df['Name'] + ' ' + df['Surname']
    df.drop(['Name', 'Surname'], axis=1, inplace=True)
    df.rename(columns={'Id': 'AvatarId'}, inplace=True)
    return df
# }}}


# {{{ REGION: Users
@lru_cache(maxsize=100)
def get_users_anag():
    print('*** Loading users anag from csv')
    df = pd.read_csv(co.dumps_path()+'out_qAbpUsers.csv', sep='$')
    df['NameSurname'] = df['Name'].str.title() + ' ' + \
        df['Surname'].str.title()
    df = df[['Id', 'ClientCode', 'PdcCode', 'NameSurname']]
    df = df.rename(columns={'Id': 'UserId'})
    df = df[~df.ClientCode.isnull()]
    df['UserId'] = df.UserId.astype(int)
    return df


@lru_cache(maxsize=100)
def get_customers_anag():
    print('*** Loading customers anag from csv')
    df = pd.read_csv(co.dumps_path()+'out_qVrCustomers.csv', sep='$')
    return df


@lru_cache(maxsize=100)
def get_province():
    print('*** Loading province from csv')
    df = pd.read_csv(co.dumps_path()+'out_qProvince.csv', sep='$')
    return df


@lru_cache(maxsize=100)
def get_regions():
    print('*** Loading regions from csv')
    df = pd.read_csv(co.dumps_path()+'out_qRegioni.csv', sep='$')
    return df


@lru_cache(maxsize=100)
def get_user_roles():
    print('*** Loading user roles from csv')
    df = pd.read_csv(co.dumps_path()+'out_qAbpUserroles.csv', sep='$')
    return df

# }}}


# {{{ REGION: Web log
@lru_cache(maxsize=100)
def get_web_log(uid):
    print('*** Loading web log for user [{0}]'.format(uid))
    if uid == 2668:
        df = pd.read_csv(co.dumps_path()+'Ferrari.csv', sep='$')
    elif uid == 3607:
        df = pd.read_csv(co.dumps_path()+'Cataldi.csv', sep='$')
    else:
        logging.warning("No web log available for user [{0}]"
                        .format(uid))
        return None

    df = df[df.Azione.str.contains('analytic', False)]
    df.Tempo = pd.to_datetime(df.Tempo, dayfirst=True).dt.floor('D')
    df = df.Tempo.unique()
    return df
# }}}


# {{{ REGION: Sessions
@lru_cache(maxsize=100)
def get_avatar_history():
    # Questa tabella ha molte più info ma per ora mi serve sessiondate
    print('*** Loading sessions history from csv')
    df = pd.read_csv(co.dumps_path()+'out_qVrAvatarHistory.csv', sep='$')
    df = df[['UserId', 'SessionId', 'AvatarId', 'StartDate', 'Total']]
    df.StartDate = pd.to_datetime(df.StartDate, dayfirst=True)
    df = dm.add_aggregate_date(df, 'StartDate')
    df = df.drop('StartDate', axis=1)
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
