#!/usr/bin/env python

import pandas as pd
from DA import CsvLoader as CsvL


def merge_avatar(df, cut_pce=[]):
    avatar_pce = get_avatar_pce()
    if len(cut_pce) > 0:
        avatar_pce = avatar_pce[~avatar_pce.AvatarPce.isin(cut_pce)]
    df = pd.merge(df, avatar_pce, on=['AvatarId', 'SessionId'], how='left')
    df = df[~df.AvSessId.isnull()]
    return df


def get_avatar_pce():
    avatar_pce = CsvL.get_avatar_pce()
    av_anag = CsvL.get_avatar_anag()
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
