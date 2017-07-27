#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np


def csv_reader(path):
    df = pd.read_csv(path, na_values=['0'])

    df.Date = pd.to_datetime(df.Date)
    df.Start = pd.to_timedelta(df.Start)
    df = df.sort_values(by=['Date', 'Start'])  # sort by date, and start time,

    x_date = df.Date
    x_weekday_not = (df.Date.dt.weekday <= 4).astype(int)
    x_vehicle = df.Vehicle.astype(
        'category').cat.codes.values  # categorical, to be encoded into 1-hotkey encode in preprocessing if needed
    x_service = df.Service.astype('category').cat.codes.values  # categorical, irrelevant, not included in analysis
    x_start = (df.Start / np.timedelta64(1, 's')).astype(
        int)  # translate into seconds (start - 0:00:00), to be used in regression
    x_duration = pd.to_timedelta(df.Duration) / np.timedelta64(1,
                                                               's')  # in seconds, for all the cases(End-Start)=Duration. End is dropped.
    x_spans = df[['Stop1', 'Stop1to2', 'Stop2', 'Stop2to3', 'Stop3', 'Stop3to4',
                  'Stop4', 'Stop4to5', 'Stop5', 'Stop5to6', 'Stop6', 'Stop6to7',
                  'Stop7', 'Stop7to8', 'Stop8', 'Stop8to9', 'Stop9', 'Stop9to10',
                  'Stop10', 'Stop10to11', 'Stop11', 'Stop11to12', 'Stop12',
                  'Stop12to13', 'Stop13', 'Stop13to14', 'Stop14', 'Stop14to15',
                  'Stop15', 'Stop15to16', 'Stop16', 'Stop16to17', 'Stop17',
                  'Stop17to18', 'Stop18', 'Stop18to19', 'Stop19', 'Stop19to20', 'Stop20', 'Stop20to21']].as_matrix()

    return (x_date, x_vehicle, x_service, x_start, x_duration, x_spans, x_weekday_not)
