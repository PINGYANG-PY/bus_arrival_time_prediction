#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import division


import numpy as np
import statsmodels.tsa.api as smt


from detrend import  day_diff,reverse_daydiff



"""
ARIMA model fitting

FIXME: typical problem of "Estimating same model over multiple time series"

"""


def model_select_fit(y, order,VERBOSE):
    """
    The original signals and the detrending residuals are modeled using ARIMA(p,d,q) models. The best model is selected by
    grid searching around the pre-chosen (p,_,q) while keeping d fixed. The AIC is used to compare models with the same d.

    Asymptotically, minimizing the AIC is equivalent to minimizing the out-of-sample one-step forecast MSE for time series models.

    """
    best_aic = np.inf
    best_mdl = None

    p = order[0]
    d = order[1]
    q = order[2]

    p_rng = range(p, max(-1, p - 3), -1)
    q_rng = range(q, max(-1, q - 3), -1)


    trend='nc'
    # c will be very close to zero for most detrended or differenced variables.

    for i in p_rng:
        for j in q_rng:
            try:
                tmp_mdl = smt.ARIMA(y, order=(i, d, j)).fit(method='mle', trend=trend,disp=VERBOSE)
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_mdl = tmp_mdl
            except:
                continue

    return d,best_mdl



def armapred_1(y, p, q, param):
    # 1-step-ahead forecast for ARMA(p,q)
    # simplified just to  illustruation the process,
    # can use a internal function _arma_predict_out_of_sample from TSA package to do "forecast without updating (essentially filtering)"
    # see the code: http://www.statsmodels.org/dev/_modules/statsmodels/tsa/arima_model.html

    # In burn-in period, set forecasts= observation values, error term =0
    # TODO: in practice, burn-in period should not be handled more carefully
    burn_in = max(p, q)
    y_hat = y[0:burn_in]
    e = np.zeros(burn_in)


    for i in range(burn_in, y.shape[0]):

        x_ar = y[i- p:i]
        x_ma = e[i - q:i]
        x = np.concatenate((x_ar, x_ma))

        y_hat = np.append(y_hat, np.dot(param, x))

        if y[i]==np.nan: y[i]=y_hat[i]
        e = np.append(e, y_hat[i] - y[i])

    return y_hat




def predict_out(ts, d, mdl ):
    """
    Generate 1-step-ahead out-of-sample prediction using model mdl, on new observation series y.

    Here it is assumed the process can be describe by a static model, and thus omit the online model update step. The procedure is as below:
        1. Select order and estimate parameters for mdl using training dataset, and fix the model
        2. Use the fitted model to get out-of-sample predictions conditional on the observations in the test dataset

    TOD0: Alternatively, the model can be updated dynamically in three steps:
    1. forecast (m steps)]
        => y_hat[t..t+m]=mdl.predict
    2. compare forecasts with new observations and calculate evaluation metrics
        => y[t..t+m] - y_hat[t..t+m]
    3. include the new observation in training dataset and update the model
        => training_data.append(y[t..t+m])   model = sm.tsa.ARMA(y, orders)) results = model.fit()
    Similiary, the travel/stop time for the most recent bus can also be used in the EWMA filtering.


    The forecasting is performed on a daily base to avoid using data from the previous day to forecast the early morning data.

    :param d: order of differencing
    :param mdl : fitted model
    :param y: observation series with Datetimeindex

    :return: 1-step-ahead forecasts, with Datetimeindex
    """

    if mdl is None and d ==0: return  ts*0

    y= day_diff(ts,d)
   
    p = mdl.k_ar
    q = mdl.k_ma
    param = mdl.params
    # y=y.dropna()
    y_hat=y.groupby(y.index.date).transform(lambda x:armapred_1(x.values, p, q, param))

    ts_hat= reverse_daydiff(ts, y_hat,d)

    return ts_hat




