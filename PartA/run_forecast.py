
from __future__ import division

import matplotlib as mpl
#set the backend to file instead of GUI. This has to be put before importing matplotlib.pyplot.
mpl.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
import pandas as pd
import numpy as np
import logging

from reader import csv_reader,create_folders
from detrend import  rmvoutliers_fill, trend_lookup
from fit_predict import predict_out

import setting as s
VERBOSE = 1
ALPHA2=0.15

def plot_bar_chart(means, stds, title, ticklabels, legend):
    means=np.array(means)
    stds=np.array(stds)
    width = 0.1 # the width of the bars
    rects=[]
    fig, ax = plt.subplots(figsize=(15,5))
    for i in range(0, means.shape[0]):

        N = means.shape[1]
        ind = np.arange(N)  # the x locations for the groups
        rects.append(ax.bar(ind+i*width, means[i,:], width, yerr=stds[i,:]))

    # add some text for labels, title and axes ticks
    # ax.set_ylabel('')
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(ticklabels)
    ax.legend(legend)
    # plt.ylim((1,7))
    plt.show()

    return ax


def forecast(df, field_names, mdlname = None):

    if mdlname is None: return
    yy_hat=pd.DataFrame(None, columns=field_names)

    if  mdlname is 'ARIMA':
    
        with  open('./output/pkl/models_0.pkl', 'rb') as f:
            dummy_means = pd.read_pickle(f)
            d_0 = pd.read_pickle(f)
            models_0 = pd.read_pickle(f)
    
        for f in field_names:
            y = df[f]
            y = y - dummy_means[f]
            yy_hat[f] = predict_out(y, d_0[f], models_0[f]) +dummy_means[f]
    
    
    elif mdlname is 'ARIMA_trend':
    
        with  open('./output/pkl/models_trend.pkl', 'rb') as f:
            trends = pd.read_pickle(f)
            RESAMPLING_INTERVAL = pd.read_pickle(f)
            d_trend = pd.read_pickle(f)
            models_trend = pd.read_pickle(f)
              
        for f in field_names:
            y = df[f]
            y_trend = trend_lookup(y, trends[f], idx=df['Start'], resample_interval=RESAMPLING_INTERVAL)  
            resid=y - y_trend
            yy_hat[f]=y_trend+predict_out(resid, d_trend[f], models_trend[f] )
            
    elif mdlname is 'average_trend':

        with  open('./output/pkl/models_trend.pkl', 'rb') as f:
            trends = pd.read_pickle(f)
            RESAMPLING_INTERVAL = pd.read_pickle(f)

        for f in field_names:
            yy_hat[f] = trend_lookup(df[f], trends[f], idx=df['Start'],
                                   resample_interval=RESAMPLING_INTERVAL) 

    elif mdlname is 'ewma_trend':

        with  open('./output/pkl/models_trend.pkl', 'rb') as f:
            trends = pd.read_pickle(f)
            RESAMPLING_INTERVAL = pd.read_pickle(f)

        for f in field_names:
            y = df[f]
            y_trend = trend_lookup(y, trends[f], idx=df['Start'],
                                   resample_interval=RESAMPLING_INTERVAL) 
            resid = y - y_trend
            yy_hat[f]=y_trend+resid.groupby(resid.index.date).apply(lambda x: x.ewm(alpha=ALPHA2).agg('mean').shift(1))


    elif mdlname is 'ewma':
        for f in field_names:
            y = df[f]
            yy_hat[f] = y.groupby(y.index.date).apply(lambda x: x.ewm(alpha=ALPHA2).agg('mean').shift(1))


    elif mdlname is 'dummy_means':

        with  open('./output/pkl/models_dummy.pkl', 'rb') as f:
            dummy_means = pd.read_pickle(f)
            yy_hat =dummy_means
        


    return yy_hat




def main():

    """
    The performance of 5 models are compared:
        (1) ARIMA,
        (2) ARIMA with average day trend template,
        (3) day trend average,
        (4) day trend + ewma filtered detrending residual
        (5) dummy mean
    The results are save in a log file.

    """
    # Outliers are removed before being used for forecasting.
    # But outliers are included in evaluation, so the forecasts are compared with the real raw data.


    create_folders('output', ['forecasts'])
    # turn on logging
    fname = './output/' + "forecasts" + datetime.now().time().strftime("%Hh%Mm") + ".txt"
    logging.basicConfig(filename=fname, level=logging.INFO, format='%(message)s')

    df_test = csv_reader(s.filename, field_names=s.field_names, desc=s.desc_names,split=s.split-1)


    df_test_original=df_test.copy()  # keep the original data for performance evaluation

    # cleaned observations for forecasting
    # now rmvoutliers_fill is on an individual day series, while in training phase, it's on an aligned series of all the historical data.
    # use ewm with a larger ALPHAl, or a moving window with smaller window
    for f in s.field_names:
        rmvoutliers_fill(df_test[f] , fill='ewm', n=1)


    mdl_list=['ARIMA', 'ARIMA_trend', 'average_trend', 'ewma_trend', 'ewma', 'dummy_means']
    err_mae=pd.DataFrame(None, mdl_list, s.field_names)
    err_quant= pd.DataFrame(None, mdl_list, s.field_names)
    for mdl in mdl_list:
        df_hat=forecast(df_test, s.field_names, mdlname=mdl)
        # mean absolute error and 75% quantile
        err_mae.loc[mdl, :]= (df_test_original[s.field_names]-df_hat).abs().mean()
        err_quant.loc[mdl,:] = (df_test_original[s.field_names]-df_hat).abs().quantile(q=0.75)


    if VERBOSE>=1:
        # plot an example day series for all the stops.
        mdl_list2=['ARIMA_trend','average_trend','ewma_trend']
        ts=pd.DataFrame(None,columns= mdl_list2)

        date='2016-07-22'

        for f in s.field_names:
            ts['Orignal']=df_test_original.loc[date,f]
            for mdl in mdl_list2:
                ts[mdl] =  forecast(df_test[date], [f], mdlname=mdl)

            ax=ts.plot(title="Stop-travel time forecasting for " + f + ' ('+ date+')')

            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.tight_layout()

            fig = ax.get_figure()
            fig.savefig('./output/forecasts/' + f + '.pdf')
            plt.close(fig)



    logging.info('\n'*2)
    logging.info('======================================================================================================')
    logging.info('         Performance Evaluation of Stop/travel Time Forecast for Next Bus (in seconds)                   ')
    logging.info('======================================================================================================')


    logging.info('\n'*2)
    logging.info('                                   Mean Absolute Error                                          ')
    logging.info('--------------------------------------------------------------------------------------------------')
    logging.info (err_mae.to_string(line_width=100))
    logging.info('\n'*2)
    logging.info('The average mean absolute error over all the stops:\n')
    logging.info(err_mae.mean(axis=1))



    logging.info('\n'*2)
    logging.info('                           75% Quantile of Forecasting Error                                   ')
    logging.info('--------------------------------------------------------------------------------------------------')
    logging.info (err_quant.to_string(line_width=100))
    logging.info('\n'*2)
    logging.info('The average 75% error quantile over all the stops:\n')
    logging.info(err_quant.mean(axis=1))



    i=0
    k=10
    methods=['average_trend', 'ewma_trend', 'dummy_means']
    while i*k<s.field_names.__len__():
        ax=plot_bar_chart(err_mae.loc[methods].values[:,i*k:i*k+k],err_quant.loc[methods].values[:,i:i+k], 'Absolute Forcast Error and 75% Quantile', s.field_names[i*k:i*k+k],legend=methods)
        i+=1
        fig = ax.get_figure()
        fig.savefig('./output/forecast_error_'+str(i)+'.pdf')
        plt.close(fig)


    logging.info('\n'*2)
    logging.info ("""\n ----------------------    The testing is over!    ------------------------------  """)

    logging.shutdown()




if __name__ == "__main__":

    main()