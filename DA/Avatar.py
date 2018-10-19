#!/usr/bin/env python

import pandas as pd


def merge_avatar(df):
    avatar_pce = get_avatar_pce()
    df = pd.merge(df, avatar_pce, on=['AvatarId', 'SessionId'], how='left')
    df = df[~df.AvId.isnull()]
    return df


def get_avatar_pce():
    avatar_pce = pd.read_csv('./Dataset/Dumps/mail_AvatarPce.csv')
    avatar_pce.rename(columns={'Id': 'AvId'}, inplace=True)
    avatar_pce.rename(columns={'PCEId': 'AvatarPce'}, inplace=True)
    return avatar_pce


def pce_descr(pce_id):
    pce_dic = {}
    pce_dic[1] = "Stomaco"
    pce_dic[2] = "Intestino"
    pce_dic[3] = "Emorroidi"
    pce_dic[4] = "Diarrea"
    pce_dic[5] = "IBS"
    return pce_dic[pce_id]
