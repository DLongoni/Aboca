#!/usr/bin/env python

# 14-9-18: salvo questo file perchè è quello che ha generato i cluster della prima mail girata, nel caso in cui servisse replicare

# {{{ Import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# }}}

def print_outliers(df_outliers, df_plot, cols = None):
    if cols is None:
        cols = df_outliers.columns
    for col in cols:
        if False: # seleziono tipo di analisi outliers
            Q1 = np.percentile(df_outliers[col], 25)
            Q3 = np.percentile(df_outliers[col], 75)
            step = (Q3 - Q1) * 1.5
            print("Data points considered step-outliers for the col '{}':".format(col))
            display(df_plot[~((df_outliers[col] >= Q1 - step) & (df_outliers[col] <= Q3 + step))])
        else:
            Qmin = np.percentile(df_outliers[col], 2)
            Qmax = np.percentile(df_outliers[col], 98)
            print("Data points considered 2-percent outliers for the col '{}':".format(col))
            display(df_plot[~((df_outliers[col] >= Qmin) & (df_outliers[col] <= Qmax))])

def print_relevance(df, cols = None):
    if cols is None:
        cols = df.columns
    for col in cols:
        new_data = df.drop(col, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(new_data, df[col], test_size=0.25, random_state=1)
        regressor = DecisionTreeRegressor(random_state=1)
        regressor.fit(X_train, y_train)
        score = regressor.score(X_test, y_test)
        print('Variable ', col,' predictability: ', score)

