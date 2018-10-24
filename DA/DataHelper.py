#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from DA import CsvLoader as CsvL
from IPython import embed

# Livello logico tra accesso csv e data access specifici


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
