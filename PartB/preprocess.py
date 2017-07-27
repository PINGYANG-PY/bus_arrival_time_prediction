#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  data preprocessing and model selection
# 1. mark outliers as missing data
# 2. 1 hotkey encode catergorical variable
# 3. split the dataset into training set and test set:

import numpy as np


def preprocess(x_date, x_start, x_duration, x_spans, x_weekday_not):
    # x_vechicle is not selected.
    # from sklearn.preprocessing import LabelBinarizer
    # x_vehicle_1hotcode = LabelBinarizer().fit_transform(x_vehicle)

    # outlier marked as NaN
    np.seterr(invalid='ignore')
    x_spans[(x_spans - np.nanmean(x_spans, axis=0)) > 2.5 * np.nanstd(x_spans, axis=0)] = np.nan

    # if there is only one (stop/traveling time) missing, it will be fixed by calculating (total_span - sum of the avaialbe stop/traveling time)
    # otherwise left as NaN. The missing arrival time won't be used for model training
    # 141 missing data were filled
    for i in range(x_spans.shape[0]):
        if np.isnan(x_spans[i,]).sum() == 1:
            x_spans[i, np.isnan(x_spans[i,])] = x_duration[i] - np.nansum(x_spans[i,])

    # I assume that stop time and traveling are indpendant of each other, so the stop time and the traveling time are separated.
    #  The first stop stop1 is dropped, since there is no need to predict the waiting time for the first stop,
    #  and neither does it carry much information for predicting the following stops.
    #  split the data at test_end, into training set and testing set  (2/3 for training, 1/3 for testing)
    x_stops = x_spans[:, ::2]
    x_travels = x_spans[:, 1::2]
    x_stops = np.delete(x_stops, 0, 1)

    #  since data is sequential, it is more reasonable to divide the data before the first bus of the day
    t = 2 * x_spans.shape[0] // 3
    prev = x_date[t]
    for i in range(t, x_spans.shape[0]):
        if x_date[i] != prev:
            trainset_end = i
            break

    # split the data into training set (set1) and test set (set2)
    x_weekday_not1 = x_weekday_not[0:trainset_end]
    x_start1 = x_start[0:trainset_end]
    x_stops1 = x_stops[0:trainset_end, :]
    x_travels1 = x_travels[0:trainset_end, :]
    x_date1 = x_date[0:trainset_end]

    x_weekday_not2 = x_weekday_not[trainset_end:]
    x_stops2 = x_stops[trainset_end:, :]
    x_travels2 = x_travels[trainset_end:, :]
    x_start2 = x_start[trainset_end:]
    x_data2 = x_date[trainset_end:]

    return x_weekday_not1, x_start1, x_stops1, x_travels1, x_date1, x_weekday_not2, x_stops2, x_travels2, x_start2, x_data2
