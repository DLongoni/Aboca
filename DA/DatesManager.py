#!/usr/bin/env python

import pandas as pd

min_date = pd.Timestamp(2017,11,1)
max_date = pd.Timestamp(2018,8,1)

def filter_date(df, date_column):
    df = df[df[date_column] >= min_date]
    df = df[df[date_column] < max_date]
    return df

def add_aggregate_date(df, date_column):
    df['YearMonth'] = df[date_column].dt.to_period('M') # ragiono per mesi
    df['YMD'] = df[date_column].dt.to_period('D') # ragiono per mesi
    return df

