#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def day_diff(y, order=1):
    for i in range(order):
        y=y.groupby(by=y.index.date).transform(pd.Series.diff)
    return y



#get the 1-step forecast from d-order deference , there is close-form but iterative process is used for flexiblity.
def reverse_daydiff(ts, ds, order=1):
    if order==0: return ds
    diff = [None]*order
    diff[0]=ts
    for i in range(1, order):
        diff[i] = diff[i - 1].groupby(by=diff[i-1].index.date).transform(pd.Series.diff)

    for i in range(order - 1, -1, -1):
        ds = diff[i].groupby(diff[i].index.date).shift(1) + ds
    return ds





def align(df):
    """
    Align samples from different days in the whole training set according to their Start time without considering the date,
    so that an average trend can be extracted by filtering. The function is inplace.

    :param df: pd.Dateframe
    :return: pd.Dateframe with a new datetimeindex and the old dateimeindex copied to column Datetime.
    """

    df['Datetime'] = df.index
    df.set_index(df['Start'], drop=False, inplace=True)
    df.sort_index(inplace=True)


def mapback(df):
    """
    Map the data back to it's orignal Datetimeindex. The function is inplace.
    :param df (pd.Dateframe)
    :return (pd.Dateframe)

    """
    df.set_index(df['Datetime'], drop=False, inplace=True)  #keep for later use
    df.sort_index(inplace=True)



def smoothing(y, filter='ewm', a = 0.1):
    """
    batch smoothing the time series, using:

    :param y(pd.Series): time series to be processed
    :param filter(str): 'ewm', or 'moving window'
    :param a (float): smoothing factor for EWMA smoother
    :return: tuple (moving mean, moving std)
    """
    if filter == 'moving window':
        # Set min_periods to much less than window size to fill up the missing data
        #would be nice if we can use y.rolling(window='60s'), but "center" is not implemented for datetimelike and offset based windows
        winsize=int(y.resample('5min').count().mean())
        roll = y.rolling(window=winsize, center=True, min_periods=5)

    else :
        # EWM method expands the previous smoothing result so it fills up the missing data automatically.
        roll = y.ewm(alpha=a)

    m = roll.agg(['mean', 'std'])

    return m['mean'],m['std']







def rmvoutliers_fill(y, n=1, fill='moving mean'):

    """
    remove outliers and fill up missing/outlier points; inplace change
    :param y(pd.Series): : time series to be processed,
    :param n(unsigned): repeat the smoothing-removing procedure n times
    :return: pd.Series

    """
    # TODO: settingwithcopy warning, because y=df[field] in the caller. df.iloc[indexing condition, field] should be used to avoid warning.

    for i in range(n):
        ts_mean,ts_std = smoothing(y)
        y.loc[np.abs(y - ts_mean) > 2 * ts_std] = np.nan # mark the outliers as NaN,

    # Fill up the missing data by smoothing the align series. Note that moving window with small in_period or EWMA filter can
    # interpolate a signal. Since align series is used, this procedure is equivalent to filling with the (wighted) average of
    # all training data for a same stop/span in the surrounding time window.

    if fill=='moving mean' :
        y_filled, _ = smoothing(y, filter='moving window')
        where_nan =y.isnull()
        y.loc[where_nan] = y_filled.loc[where_nan]


    elif  fill=='ewm':
        y_filled, _ = smoothing(y, filter='ewm')
        where_nan =y.isnull()
        y.loc[where_nan] = y_filled.loc[where_nan]


    return y



def exclude_naday(y, naratio=0.1):
    # Drop day residual series with too many missing data/outliers
    # The previous step should have filled up most missing data, if there are still many NaNs in a daily series,
    # the series for the whole day should be excluded for modeling.

    # note: split-apply-combine create new object. Group chunks should be treated as immutable

    y = y.groupby(y.index.date).transform(lambda y1: y1 * np.nan if y1.isnull().sum() / y1.shape[0] > naratio
                            else rmvoutliers_fill(y1, fill='ewm'))
    return y



def get_avg_trend(y, filter='ewm', a=0.015, verbose =1, resample_interval='60s', fill_missing=False, title= '' , note= ''):
    """
    Get the average daily trends for all the fields,  with respect to the bus Start time.

    EWMA is much better at smoothing out the short-term variation than the moving average filter. However, EWMA filter always
    cause delay (shift) in the trend. A trick is used here to solve this issue: averaging a forward EWMA and reverse EWMA.

    Final step is resampling to create an daily trend template that evenly distributed over time. Here trend is
    resampled to every resample_interval=60 seconds. Considering the average bus dispatch interval is around 450 seconds.
    This resampling interval won't cause much information loss.

    The purpose is for easy look-up during training and real time prediction.
    When we need to look up for a trend value for timestamp t, t will be around to the closest resample_interval.
    For example, given t= 06:28:11, around t to 06:28:00, then look the trend template up by DatetimeIndex=06:28:00.
    This will be much faster than searching for the closest DatetimeIndex to t, as the DatetimeIndex is hashed.

    """

    # Two-way EWMA averaging
    ts_mean1, ts_std1 = smoothing(y, filter=filter, a=a)

    reversed_y = y.iloc[::-1]
    ts_mean2, ts_std2 = smoothing(reversed_y, filter=filter,a=a)
    ts_mean2 = ts_mean2.iloc[::-1]
    ts_std2 = ts_std2.iloc[::-1]

    ts_mean = (ts_mean1 + ts_mean2)/2
    ts_std = (ts_std1 + ts_std2)/2


    # Resample the daily trend by calculating the median of a resampling slice. mean can also be used.
    trend = ts_mean.resample(resample_interval).mean()
    ts_std = ts_std.resample(resample_interval).mean()

    # Fill up the missing trend samples if exist, by propagating the last valid
    if fill_missing:  #rolling filter introduce Nan at the head or tail..
        trend.fillna(method='ffill', inplace=True, limit=2)  #fill the end
        trend.fillna(method='bfill', inplace=True, limit=2)  #fill the start



    if  verbose>=1:
        t = title if title is not None else 'Average Trend'

        fig = plt.gcf()

        plt.plot(y[::1+y.shape[0]// 2000], alpha=.5)
        ax = trend.plot()
        ax.fill_between(trend.index, trend - 2 * ts_std, trend + 2 * ts_std,
                        alpha=.25)
        ax.legend(['Orignal', 'Trend', 'std'])
        plt.text(ax.get_xlim()[0], ax.get_ylim()[0] + 50, note)
        plt.title(t)
        plt.show()

        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.tight_layout()

        fig.savefig('./output/trends/'+t + '.pdf')
        plt.close(fig)

    return trend



def trend_lookup(y,trend, idx=None,  resample_interval='60s'):
    # look up the trend value by idx(datetimeindex).

    if idx is None: idx=y.index

    # Round in the Start time for the beginning and end of the day
    idx.values[idx< trend.index.min()] = trend.index.min()
    idx.values[idx> trend.index.max()] = trend.index.max()

    y_trend=y.copy() #NOTE: deep copy to avoid changes to the original series
    y_trend.iloc[:] = trend[pd.DatetimeIndex(idx).round(resample_interval)].values

    return  y_trend

