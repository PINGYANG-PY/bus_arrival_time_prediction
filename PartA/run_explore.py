from __future__ import division
from datetime import datetime
import pickle as pkl
import logging

# set backend to non-interactive to save figures without showing on screen by using fig.savefig
import matplotlib as mpl
mpl.use('Agg')

from reader import csv_reader, create_folders
from majorsteps import check_info_gain, check_bus_dispatch_interval, denoise_extract_trend, check_xcorr, get_pdq
import setting as s

VERBOSE = 1

def main():

    # create an folder "Output" under the current folder to hold results
    create_folders('output',['acf-pacf','trends','pkl'])

    #turn on logging
    fname ='./output/' + 'datastats_' + datetime.now().time().strftime('%Hh%Mm') + '.txt'
    logging.basicConfig(filename=fname, level=logging.INFO, format='%(message)s')

    df = csv_reader(s.filename, field_names=s.field_names, desc=s.desc_names, nrows=5000)


    #ALPHA need to be tuned according to training sample size
    ALPHA = s.set_alpha(df.shape[0])



    # Time-consuming!... just refer to the attached file Infogain.txt
    # logging.info('\n' * 2)
    # logging.info('===================================================================================')
    # logging.info('                       Information Gain by different variables                     ')
    # logging.info('===================================================================================')
    #
    # condits=['SlicedStart','Vehicle','Weekday','Weekend_not','Service','SlicedStart+Weekday']
    # gain, summary = check_info_gain(df, s.field_names, condits)
    #
    # logging.info("""\n We observed that :
    # \t 1. Start time is relevant, indicating day trends exist with respect to Start time
    # \t 2. Vehicle has some relevance, but likely due to outliers (some group is very small).
    # \t 3. Weekdays has little relevance.
    # \t 4. (Start, Weekdays) has some relevance, but likely due to outliers, ignored for now.\n """)



    logging.info('\n' * 2)
    logging.info('===================================================================================')
    logging.info('                         Daily bus dispatch intervals                              ')
    logging.info('===================================================================================')
    logging.info('Check the day trend plot of dispatch interval in the folder ./output/trends')

    check_bus_dispatch_interval(df,alpha=ALPHA, resample_interval=s.RESAMPLING_INTERVAL)

    logging.info("""The trend plot shows
         \t 1.the dispatch interval varies over a day.
          \t 2.The standard deviation over the smoothing window indicates that the dispatch interval varies greatly between different days.\n
    Based on the above observation and considering the randomness in bus dispatch, it does not make much sense to evenly resample the daily data over time. Instead, we just treat each bus run in a day as a sequential data point.\n""")




    logging.info('\n' * 2)
    logging.info('===================================================================================')
    logging.info('              Average Day Trends of All the Stop/Travel Times                      ')
    logging.info('===================================================================================')
    logging.info('Check the day trend plots in folder ./output/trends. The plots is downsampled for ploting')
    df, trends,df_res= denoise_extract_trend(df,  field_names=s.field_names, alpha=ALPHA, verbose=VERBOSE, resample_interval=s.RESAMPLING_INTERVAL)



    logging.info('\n' * 2)
    logging.info('===================================================================================')
    logging.info('              cross-correlation between the detrend residuals                      ')
    logging.info('===================================================================================')

    check_xcorr(df_res,s.field_names)

    logging.info('\n We observe that residuals after detrending mostly are not cross-correlated. Only few couples have low correlation. ')




    # logging.info('\n' * 2)
    # logging.info('===================================================================================')
    # logging.info('          information gain of categorical variables on residuals after detrending          ')
    # logging.info('===================================================================================')
    #
    # condits=['SlicedStart','Vehicle','Weekday','Weekend_not','Service','SlicedStart+Weekday']
    # gain_res, summary_res = check_info_gain(df_res.join(df[s.desc_names]), s.field_names, condits)
    #
    # logging.info('\n We observe that information gain Start time.')
    #



    logging.info('\n'* 2)
    logging.info('===================================================================================')
    logging.info('       Investigate p,d,q orders for ARIMA model fitting                             ')
    logging.info('===================================================================================')
    logging.info('\nThe d-order is determined by differencing+ADF test iteratively. ')
    logging.info("""Tentative p,q are chosen by investigating the autocorrelation and partial autocorrelation plots. Check the ACF-PACF plots in the folder ./output/acf-pacf""")


    orders= get_pdq(df, df_res, s.field_names)
    with open('./output/pkl/orders.pkl', 'wb') as f:
        pkl.dump(orders, f)

    logging.info('\n')
    logging.info('               ARIMA(p,d,q) order - by Acf-Pacf confidence level check                           ')
    logging.info('--------------------------------------------------------------------------------------------------')
    logging.info (orders.to_string(line_width=100))

    logging.info ("""\nFor %d out of %d variables, the detrend residuals appear as white noise. For these stops/spans, the trend
    average corresponding to the Start time will be a good prediction. """
           % ((orders.loc['residual']==(0,0,0)).sum(),orders.shape[1]))

    # TODO:The p, q identified are tentative. The p, q values should be manually altered by visually checking the shape of ACF and PACF.
    # For example, p for AR (q for MA) term should be removed if the pacf (acf) has an obvious cut-off pattern.
    # Some series has a high-order AR term, indicating under-differencing.



    logging.info('\n'*2)
    logging.info ("""\n ----------------------    The data exploration is over!    ------------------------------  """)

    logging.shutdown()




if __name__ == "__main__":

    main()


