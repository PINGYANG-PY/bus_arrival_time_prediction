from __future__ import division

#set backend to non-interactive to save figures without showing on screen by using fig.savefig
# must be called before import other module. as it has to be callled before  import matplotlib pylab
import matplotlib as mpl
mpl.use('Agg')

import pandas as pd
import pickle as pkl
import logging


from reader import csv_reader,create_folders
from majorsteps import denoise_extract_trend
from fit_predict import model_select_fit
from logger import Logger
from datetime import datetime


import setting as s
VERBOSE = 1

def main():
    
    # create an folder "Output" under the current folder to hold results, trained models are save in 'pkl'
    create_folders('output',['pkl'])
    # Write results to file, return stdout back to screen by  sys.stdout.back_to_screen() if needed.
    fname = './output/' + "models" + datetime.now().time().strftime("%Hh%Mm") + ".txt"
    logging.basicConfig(filename=fname, level=logging.INFO, format='%(message)s')

    df = csv_reader(s.filename, field_names=s.field_names, desc=s.desc_names, split=s.split)

    #ALPHA need to be tuned according to training sample size
    ALPHA = s.set_alpha(df.shape[0])

 


    #denoise, forward mode, can be altered to use online
    df, trends,df_res= denoise_extract_trend(df,  field_names=s.field_names, alpha=ALPHA, verbose=VERBOSE, resample_interval=s.RESAMPLING_INTERVAL)




    # select, estimate and pickle the ARIMA models for the original series and the detrending residuals (ARIMA with trend term)

    models_0={}
    d_0={}
    models_trend ={}
    d_trend={}

    dummy_mean=df[s.field_names].mean()
    df_zeroed=df[s.field_names]-dummy_mean

    # get tentative pdq orders
    with  open('./output/pkl/orders.pkl', 'rb') as f:
        pdq =pd.read_pickle(f)

    for name, dff,d, mdl in zip(['series', 'residual'], [df_zeroed, df_res],[d_0,d_trend],[models_0,models_trend]):
        for f in s.field_names:
            d[f], mdl[f] = model_select_fit(dff[f].dropna().values, pdq.loc[name, f],VERBOSE=VERBOSE>=2)
            if VERBOSE>=1:
                logging.info('\n'*4 + 'For {} -  {} after {}-order differencing'.format(f, name, d[f]))
                logging.info('\nNone' if mdl[f] is None else mdl[f].summary())

    # pickle all we need for the model with a trend term
    with  open('./output/pkl/models_trend.pkl', 'wb') as f:
        pkl.dump(trends, f)
        pkl.dump(s.RESAMPLING_INTERVAL, f)
        pkl.dump(d_trend, f)
        pkl.dump(models_trend, f)

    # pickle the model for the original series
    with  open('./output/pkl/models_0.pkl', 'wb') as f:
        pkl.dump(dummy_mean, f)
        pkl.dump(d_0, f)
        pkl.dump(models_0, f)

    # The dummy mean of the training set will also be used for comparing with the forecasts.
    with  open('./output/pkl/models_dummy.pkl', 'wb') as f:
        pkl.dump(dummy_mean, f)


    logging.info('\n'*2)
    logging.info('The models are save as models_trend.pkl, models_0.pkl, and models_dummy.pkl')
    logging.info ("""\n ----------------------    Model training is over!    ------------------------------  """)


    logging.shutdown()



if __name__ == "__main__":

    main()