#!/usr/bin/env python

import pandas as pd


def merge_avatar(df):
    avatar_pce = pd.read_csv('./Dataset/Dumps/mail_AvatarPce.csv')
    avatar_pce.rename(columns={'Id': 'AvId'}, inplace=True)
    avatar_pce.rename(columns={'PCEId': 'AvatarPce'}, inplace=True)
    df = pd.merge(df, avatar_pce, on=['AvatarId', 'SessionId'], how='left')
    df = df[~df.AvId.isnull()]
    return df
