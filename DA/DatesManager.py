#!/usr/bin/env python
# coding: utf-8

import pandas as pd

MINDATE = pd.Timestamp(2017, 11, 1)
MAXDATE = pd.Timestamp(2018, 9, 30)


def filter_date(df, date_column, max_date=-1):
    if max_date == -1:
        max_date = MAXDATE
    elif not isinstance(max_date, pd.Timestamp):
        max_date = pd.Timestamp(max_date)

    df = df[df[date_column] >= MINDATE]
    df = df[df[date_column] <= MAXDATE]
    return df


def add_aggregate_date(df, date_column):
    df['YearMonth'] = df[date_column].dt.to_period('M')  # ragiono per mesi
    df['YMD'] = df[date_column].dt.floor('D')  # ragiono per mesi
    return df
