#!/usr/bin/env python

import pandas as pd


def merge_avatar(df, cut_pce=[]):
    avatar_pce = get_avatar_pce()
    if len(cut_pce) > 0:
        avatar_pce = avatar_pce[~avatar_pce.AvatarPce.isin(cut_pce)]
    df = pd.merge(df, avatar_pce, on=['AvatarId', 'SessionId'], how='left')
    df = df[~df.AvSessId.isnull()]
    return df


def get_avatar_pce():
    avatar_pce = pd.read_csv('./Dataset/Dumps/out_qVrAvatarsPce.csv', sep='$')
    avatar_pce = avatar_pce[['Id', 'SessionId', 'AvatarId', 'PCEId']]
    avatar_pce.rename(columns={'Id': 'AvSessId'}, inplace=True)
    avatar_pce.rename(columns={'PCEId': 'AvatarPce'}, inplace=True)

    av_anag = pd.read_csv('./Dataset/Dumps/out_qVrAvatars.csv', sep='$')
    av_anag = av_anag[['Id', 'Name', 'Surname', 'Age']]
    av_anag['AvName'] = av_anag['Name'] + ' ' + av_anag['Surname']
    av_anag.drop(['Name', 'Surname'], axis=1, inplace=True)
    av_anag.rename(columns={'Id': 'AvatarId'}, inplace=True)
    avatar_pce = pd.merge(avatar_pce, av_anag, left_on='AvatarId',
                          right_on='AvatarId')
    return avatar_pce


def pce_descr(pce_id):
    pce_dic = {}
    pce_dic[1] = "Stomaco"
    pce_dic[2] = "Intestino"
    pce_dic[3] = "Emorroidi"
    pce_dic[4] = "Diarrea"
    pce_dic[5] = "IBS"
    return pce_dic[pce_id]
