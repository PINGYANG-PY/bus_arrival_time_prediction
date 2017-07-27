#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
from __future__ import print_function


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import acf,pacf,adfuller

from sampen.sampen2 import  sampen2



# import datastats
# reload(datastats)
# from datastats import plot_acf_pacf, plot_heatmap, get_corr_pair,test_stationarity,is_stationary,info_gain


def info_gain(df, fieldnames, keys, nlargest=2):
    """
    Evaluate whether including [keys] can achieve significant information again for [field_names] in dataframe [df].
    """

    if  type(keys) is not str:
        raise TypeError('keys must be a string')

    keys=keys.split('+')

    def entropy(y):
        return  sampen2(y.values, 0)[0][1] #or e

    en0 = df[fieldnames].agg(entropy)
    groups=df.groupby(keys)

    en=groups[fieldnames].apply(lambda y: y.shape[0]/ df.shape[0] * y.agg(entropy)).sum()

    info_g=en0-en
    infosum = info_g.sum()
    min_gsize=groups.size().min()
    topn=info_g.nlargest(nlargest).to_string(float_format='%1.3f')

    return info_g, [infosum, min_gsize,topn]




def plot_acf_pacf(y, fname='', lags=10, figsize=(10, 7), style='bmh', title='',alpha=0.05):
    """
    Plot autocorrelation function and partial autocorrelation function, and tentatively identify p,q for the ARMA model

    :param y: y should not contain NaN values
    :param alpha: critical level for determin p, q. Smaller alpha generates higher confidence level, thus smaller p, q.
    :return: possible p, q for AR and MA models
    """

    # Identify the smallest p,q in the critical range.
    # FIXME:  not accurate, need to modify to recognize seasonality.
    def det_order(x):
        c = x[0]
        confint = x[1]

        flag = False
        for i in range(1,6): #forward checking

            if (c[i] > confint[i, 0] - c[i]) and (c[i] < confint[i, 1] - c[i]):
                flag = True
                break
        return i-1 if flag == True else 0

    x = pacf(y, nlags=lags, alpha=alpha)
    p = det_order(x)

    x = acf(y, nlags=lags, alpha=alpha)
    q = det_order(x)



    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    y.dropna()

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        plt.text(acf_ax.get_xlim()[0]+1, acf_ax.get_ylim()[0]+0.5, 'q='+str(q), fontsize=14)

        pacf_ax = plt.subplot2grid(layout, (1, 1))
        plt.text(pacf_ax.get_xlim()[0]+1, pacf_ax.get_ylim()[0]+0.5,'p='+str(p), fontsize=14)

        y[::1+y.shape[0]// 2000].plot(ax=ts_ax) #todo: downsanple y for better display
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=alpha)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=alpha)

        plt.tight_layout()

    fig.savefig('./output/acf-pacf/'+fname+'.pdf')
    plt.close(fig)


    return p, q


def plot_heatmap(corr):
    # Plot corr matrix out as heatmap
    fig, ax = plt.subplots(figsize=(7, 7))
    heatmap = ax.pcolor(corr * 2, cmap=plt.cm.Blues, alpha=1)  # *2 to increase the darkness
    # Format
    fig = plt.gcf()
    ax.set_frame_on(False)
    # more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(corr.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(corr.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(corr.index, minor=False)
    ax.set_yticklabels(corr.index, minor=False)
    # rotate xticks
    plt.xticks(rotation=90)
    ax.grid(False)
    # Turn off all the ticks
    plt.gca()

    fig.savefig('./output/'+'cross-correlation.pdf')
    plt.close(fig)




def get_corr_pair(corr, fieldnames,lo,hi):
    indices = np.where((corr >= lo) & (corr < hi))

    result = pd.DataFrame(([y - x, fieldnames[x], fieldnames[y], corr.values[x][y]]
                           for x, y in zip(*indices) if x != y and x < y),
                          columns=['distance', 'field1', 'field2', 'xcorr'])

    result=result.sort_values(by=['distance'], ascending=True)

    print('\n' )
    print('Between [%.1f  %.1f) '% (lo, hi))
    print('-------------------')
    print(result.to_string(index=False))





def test_stationarity(timeseries, name=None, verbose=1, critical='5%'):
    # verbose =0: no message
    #         =1: print conclusion
    #         =2: print detailed test results

    name = 'Time Series' if name is None else name

    # remember to set maxlag for adfuller , otherwise exception "maxlag should be < nobs" will be raise if the groupsize is too small.
    dftest = adfuller(timeseries, autolag='AIC',maxlag=10)



    if verbose==2:
        print('\n'*2)

    if dftest[0] > dftest[4][critical]:
        res= False
        if verbose == 1: print (name +' is nonstationary')
    else:
        res=True
        if verbose == 1: print  (name +' is stationary')

    if verbose==2:
        print ('Results of Dickey-Fuller Test:')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        dfoutput['Critical Value 5%'] = dftest[4]['5%']
        print (dfoutput.to_string())

    return res



def is_stationary(ts,th=0.8):
    """
    Check if a variable (stop/travel time for a stop/span) is stationary.

    ADF and KPSS test are both based on unit root test, and they often mistaken seasonal signal as stationary.
    In our case, the daily series in the calendar order is a typical seasonal series. To avoid this error, we
    have to perform the ADF test on every daily series individually. If >80% daily series for a stop/span are
    stationary, it will be taken as stationary.

    95% confidence level were used in ADF test, and the variable is considered stationary if 80% daily series are stationary.
    The confidence level chosen may result in "mild underdifferencing", but it can be compensated for by adding AR terms to the model,

    """
    res = list()
    for  n,g in ts.groupby(by=ts.index.date):
        res.append(test_stationarity(g.dropna().values,  verbose=0, critical='5%'))

    return np.sum(res)/np.size(res)>=th

