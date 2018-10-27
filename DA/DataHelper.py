#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from DA import CsvLoader as CsvL
from IPython import embed

# Livello logico tra accesso csv e data access specifici


def top_prod_av_breakdown(df, av_id, nprod=10):
    df_av = df[df.AvSessId == av_id]
    # prodotti più frequentemente consigliati sbagliati a questo avatar
    df_av = rwcount_base(df_av, 'ProductId', 'Product')
    df_av = df_av.nlargest(nprod, 'nTot').reset_index().fillna(0)
    df_av = add_prod_name(df_av)
    return df_av


def sess_start(uhist):
    ustart = uhist.groupby('SessionId').min().reset_index().YMD
    return ustart


def rwcount_base(df, group, type_suffix, count_col='Id'):
    rstring = 'Right{0}'.format(type_suffix)
    wstring = 'Wrong{0}'.format(type_suffix)
    df_f = df[(df.ActionType == rstring) | (df.ActionType == wstring)]
    df_rw = df_f.groupby(group).apply(lambda x: pd.Series(
        {'Ratio': sum(x.ActionType == rstring) / x[count_col].count(),
            'RightCount': sum(x.ActionType == rstring),
            'nTot': x[count_col].count()})).reset_index()
    return df_rw


def add_prod_name(df):
    p_anag = CsvL.get_prod_anag()
    p_anag = p_anag[['ProductId', 'ProdName']]
    df = pd.merge(df, p_anag, left_on='ProductId', right_on='ProductId')
    return df


def add_session_date(df):
    ah = CsvL.get_avatar_history()
    df = pd.merge(df, ah, on=['AvatarId', 'SessionId', 'UserId'])
    return df


def add_avatar_data(df, cut_pce=[]):
    avatar_pce = CsvL.get_avatar_pce()
    av_anag = CsvL.get_avatar_anag()
    avatar_pce = pd.merge(avatar_pce, av_anag, left_on='AvatarId',
                          right_on='AvatarId')
    if len(cut_pce) > 0:
        avatar_pce = avatar_pce[~avatar_pce.AvatarPce.isin(cut_pce)]

    on_col = __av_merge_col(df)
    df = pd.merge(df, avatar_pce, on=on_col, how='left')
    df = df[~df.AvSessId.isnull()]
    return df


def pce_descr(pce_id):
    pce_dic = {}
    pce_dic[1] = "Stomaco"
    pce_dic[2] = "Intestino"
    pce_dic[3] = "Emorroidi"
    pce_dic[4] = "Diarrea"
    pce_dic[5] = "IBS"
    return pce_dic[pce_id]


def __av_merge_col(df):
    if 'AvSessId' in df.columns:
        on_col = 'AvSessId'
    else:
        on_col = ['AvatarId', 'SessionId']
    return on_col


if __name__ == "__main__":
    embed()
