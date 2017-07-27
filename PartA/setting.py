
from __future__ import division

filename = 'Route24304Train.csv'



# field_names = ['Stop1', 'Stop1to2', 'Stop2']
desc_names= [u'Vehicle', u'Service', u'Start_sec', u'Start', u'Weekday', u'Weekend_not',u'SlicedStart']
field_names = ['Stop1', 'Stop1to2', 'Stop2', 'Stop2to3', 'Stop3', 'Stop3to4',
               'Stop4', 'Stop4to5', 'Stop5', 'Stop5to6', 'Stop6', 'Stop6to7',
               'Stop7', 'Stop7to8', 'Stop8', 'Stop8to9', 'Stop9', 'Stop9to10',
               'Stop10', 'Stop10to11', 'Stop11', 'Stop11to12', 'Stop12',
               'Stop12to13', 'Stop13', 'Stop13to14', 'Stop14', 'Stop14to15',
               'Stop15', 'Stop15to16', 'Stop16', 'Stop16to17', 'Stop17',
               'Stop17to18', 'Stop18', 'Stop18to19', 'Stop19', 'Stop19to20',
               'Stop20', 'Stop20to21']
split=0.667 #2/3 for training, the remaining 1/3 for test

# Temporal resolution of the average trend.
RESAMPLING_INTERVAL = '60s'


# alpha is used in EWMA filter for extract the average trend. Smaller alpha generates smoother trend.
# alpha should be tuned to the size of training set.
def set_alpha(L):
    if L  < 3000:
        ALPHA = 0.025
    elif L  < 5000:
        ALPHA = 0.02
    elif L< 7000:
        ALPHA = 0.013
    else:
        ALPHA = 0.01
    return ALPHA