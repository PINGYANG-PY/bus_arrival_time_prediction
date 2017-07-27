#!/usr/bin/python
# -*- coding: UTF-8 -*-


from __future__ import division
import pandas as pd

from datastats import plot_acf_pacf, get_corr_pair, is_stationary,info_gain,plot_heatmap
from detrend import get_avg_trend,  rmvoutliers_fill,mapback,align, trend_lookup, day_diff, exclude_naday



def check_info_gain(df, field_names, condits):
    # Investigate the information again for all the stop/travel times achieved by conditioning on other variables.
    # divide the Start time into 2-hr slices
    gain = pd.DataFrame(None, condits, field_names, dtype=float)
    summary = pd.DataFrame(None, condits, ['Sum of Info Gain', 'Smallest group size', 'n-Largest gain'])
    for var_name in condits:
        gain.loc[var_name], summary.loc[var_name] = info_gain(df, field_names, var_name, nlargest=2)

    print('\n' * 1)
    print(' Information Gain for Stop/Travel Time')
    print('------------------------------------------------')
    print(gain.to_string(line_width=100))

    print(' Summary')
    print('------------------------------------------------')
    print('\n' * 1)
    # Very small group size means that information again is achieved by outliers.
    print(summary.to_string(line_width=100))

    return gain, summary


def check_bus_dispatch_interval(df,alpha, resample_interval):
    # Investigat the bus dispatch intervals and determine if the data need to be resampled to be evenly distributed over time in a day.

    # groupby date, so the overnight gap will be marked as NaN and won't be included for analysis
    # df.insert(0, 'Interval',day_diff(df['Start_sec']))
    df['Interval']=day_diff(df['Start_sec'])
    # data from different days are aligned according to Start time
    align(df)
    # remove outliers
    rmvoutliers_fill(df['Interval'])
    # plot the average bus dispatch interval variations over a day
    m = int(df['Interval'].mean())
    std = df['Interval'].std()
    msg = ' The mean interval between bus runs is %d .' % int(df['Interval'].clip(m - 2 * std, m + 2 * std).mean())
    get_avg_trend(df['Interval'], filter='ewm', a=alpha, title='Bus Dispatch Intervals Over a Day',
                  note=msg, resample_interval=resample_interval, fill_missing=True)
    mapback(df)
    df.drop(['Interval'],axis=1,inplace=True)

    print ('\n' + msg + '\n')





def denoise_extract_trend(df, field_names, alpha, resample_interval='60s',verbose=1):
    """
    # Extract the average day trend and return the detrending residual for analysis and modeling

    :param alpha: controls the smoothness of the average trend
    :param resample_interval: downsamplien interveral for the average trend

    :return: df with outlier removed and missing data filled, average day trends as {field: pd.Dataframe}, residuals as df_res[field_names]
    """

    align(df)

    trends = {}
    res_names = [f + '_res' for f in field_names]
    df_res = pd.DataFrame(columns=field_names)
    for f in field_names:
        y = df[f]
        # remove outliers and fill them up with moving mean. This function is in place, so df in the caller will be cleaned.
        rmvoutliers_fill(y,n=1)

        trends[f] = get_avg_trend(y, filter='ewm', a=alpha, title='Trend of ' + f,
                                  resample_interval= resample_interval,
                                  fill_missing=True, verbose=verbose)

        # now remove highly corrupted day series, because they are useful for average trend extraction, but no good for daily residual modeling
        y = exclude_naday(y,naratio=0.2)

        df[f + '_res'] = y - trend_lookup(y, trends[f], resample_interval=resample_interval)

    mapback(df)

    df_res[field_names]=df[res_names]  #df_res is just a view of df[res_names]

    return df, trends, df_res



def check_xcorr(df,field_names):

    corr=df[field_names].corr().abs()
    plot_heatmap(corr)

    get_corr_pair(corr,field_names, 0.5, 1.0)
    get_corr_pair(corr,field_names, 0.4, 0.5)
    get_corr_pair(corr,field_names, 0.3, 0.4)



def get_pdq(df, df_res, field_names):

    orders = pd.DataFrame(None, ['series', 'residual'], field_names, dtype=int)

    th = 0.8
    for f in  field_names:
        for name, y in zip(['series', 'residual'], [df[f], df_res[f]]): # y: a view of df_res[f], id(y)=id( df_res[f])
            d = 0
            while (not is_stationary(y, th)) and d<=2:
                y = day_diff(y)         # y: split-apply-combine in day_diff() create a new object, id(y)!=id( df_res[f])
                d+=1

            fname = f + ' ' + name
            title = fname + ' d = ' + str(d)
            if d != 0: title = title + ' difference'
            p, q= plot_acf_pacf(y.dropna().values, fname, title=title, lags=20, alpha=0.0005)

            orders.loc[name, f] = (p,d,q)

    return orders


