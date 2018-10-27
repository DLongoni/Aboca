#!/usr/bin/env python

# {{{ Import
from IPython import embed
from DA import Users
from DA import DatesManager as dm
from DA import CsvLoader as CsvL
from DA import DataHelper as dh
# }}}


def get_df(max_date=-1):
    df = CsvL.get_avatar_info()
    df = dh.add_avatar_data(df, cut_pce=[5, 6, 7])
    df = dh.add_session_date(df)
    df.drop(['AvatarId'], axis=1, inplace=True)
    df = dm.filter_date(df, 'YMD', max_date)
    df = Users.merge_users_clean(df)
    # filtro colonne per primo test sensibilita
    df = df.drop(['Age', 'YearMonth', 'ClientCode', 'FarmaId', 'Latitudine'],
                 axis=1)
    return df


def get_user_history(user_id):
    df = get_df()
    user_hist = df[df.UserId == user_id]
    return user_hist


if __name__ == '__main__':
    embed()
