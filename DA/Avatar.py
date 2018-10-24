#!/usr/bin/env python

import pandas as pd  # NOQA
from DA import CsvLoader as CsvL  # NOQA


# def merge_avatar(df, cut_pce=[]):
#     avatar_pce = get_avatar_pce()
#     if len(cut_pce) > 0:
#         avatar_pce = avatar_pce[~avatar_pce.AvatarPce.isin(cut_pce)]
#     df = pd.merge(df, avatar_pce, on=['AvatarId', 'SessionId'], how='left')
#     df = df[~df.AvSessId.isnull()]
#     return df
#
#
# def get_avatar_pce():
#     avatar_pce = CsvL.get_avatar_pce()
#     av_anag = CsvL.get_avatar_anag()
#     avatar_pce = pd.merge(avatar_pce, av_anag, left_on='AvatarId',
#                           right_on='AvatarId')
#     return avatar_pce
#
