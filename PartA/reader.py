#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
import os



def split_data(df, split=None):

    """
    split df into [df1|df2]. The splits should be at the end of a day and not divide a day-series into two different sets.

    split   = 0.25  => [25% | 75%]
            = -0.25 => [75% | 25%]
    """

    if split == None or split>=1 or split==0: return df, None
    if split<=-1: return None, df
    if split<0: split = 1+split

    n = int(np.floor(df.shape[0]*split))

    df1 = df.iloc[0:n]  # df1: new object
    df1_tail = df.iloc[(df.index.date == df1.tail(1).index.date) & (df.index.time > df1.tail(1).index.time)]
    df1 = pd.concat([df1, df1_tail])
    df2=df.iloc[df1.shape[0]:]

    return df1, df2


def csv_reader(path=None, field_names = None,desc=None, nrows=None, split=None):


    if not os.path.isabs(path):
            path = os.path.join(os.path.curdir, path)
    df = pd.read_csv(path, na_values=['0'], nrows=nrows)
    field_names = field_names or  df.columns[7:].tolist()


    df['Datetime']=pd.DatetimeIndex(df['Date'] + ' ' + df['Start'])
    df.set_index(df['Datetime'], drop=True, inplace=True)
    df.sort_index(inplace=True)

    # fake date, used later for easy data alignment and rolling in average trend extraction, and trend looking up in forecast
    df['Start_sec']=pd.to_timedelta(df['Start']).dt.total_seconds().astype(int)
    df['Start'] = pd.DatetimeIndex('2017-01-01' + ' ' + df['Start'])

    # split after datetimeindexing,if required
    df, df2 =split_data(df, split)
    if split is not None and split<0: df = df2


    df['Weekday'] = df.index.weekday_name
    df['Weekend_not']=(df.index.weekday<= 4).astype(int)
    #divide the Start time into 2-hr slices so we can investigate the associate info gain
    df['SlicedStart'] = df.index.hour // 2

    # Duration in seconds, for all the cases(End-Start)=Duration.
    df.Duraton = pd.to_timedelta(df.Duration).dt.total_seconds().astype(int)
    # FIXME: drop irrelevant columns
    df.drop(['Date','End','Route','Duration'],axis=1,inplace=True)



    # preliminary outlier removal, just to remove very obvious outliers before calculating information gain
    def remove_outlier(df, field_names):
        for field in field_names:
            mean=df[field].mean()
            std=df[field].std()
            outlier_mask = np.abs(df[field] -mean ) > 3 * std
            df.loc[outlier_mask,field] = np.nan

    remove_outlier(df,field_names)


    return df[desc+field_names]




def create_folders(folder, subfolders):

    output_path = os.path.join(os.path.curdir, folder+'/')

    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except OSError as exc:  # Guard against race condition
            raise

    for folder in subfolders:
        acf_path = os.path.join(output_path, folder+'/')
        if not os.path.exists(acf_path):
            try:
                os.makedirs(acf_path)
            except OSError as exc:  # Guard against race condition
                raise


