#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.metrics import mean_absolute_error
import cPickle


# load the pickled models and evaluate the predicted the results on the test set

def test(x_weekday_not2, x_start2, x_stops1, x_travels2, x_date2):
    M = 2
    L = 1

    X_left = np.column_stack((x_start2, x_weekday_not2[:, np.newaxis]))
    Y_left = x_travels2[:, 0:L]

    X_stop_left = np.column_stack((x_start2, x_weekday_not2[:, np.newaxis]))
    Y_stop_left = x_stops1[:, 0:L]

    # for left start part,  drop the missing data
    mask = np.isnan(Y_left).sum(axis=1) > 0
    X_left = X_left[~mask, :]
    Y_left = Y_left[~mask, :]

    mask = np.isnan(Y_stop_left).sum(axis=1) > 0
    X_stop_left = X_stop_left[~mask, :]
    Y_stop_left = Y_stop_left[~mask, :]

    # load the model again
    with open('model_travel_left.pkl-2', 'rb') as fid:
        model_travel_left = cPickle.load(fid)
    MAE_travel_left = mean_absolute_error(Y_left, model_travel_left.predict(X_left))

    with open('model_stops_left.pkl-2', 'rb') as fid:
        model_stops_left = cPickle.load(fid)
    MAE_stops_left = mean_absolute_error(Y_stop_left, model_stops_left.predict(X_stop_left))

    MAE_travel_top = []
    MAE_travel_main = []
    MAE_stops_main = []
    MAE_stop_top = []

    # now, train the models for  stops/spans from L to terminal
    for j in range(L, x_travels2.shape[1] - 2):
        # traveling times during stops
        X_top = np.empty((0, 2 + L), int)
        Y_top = np.empty((0, 3), int)

        X = np.empty((0, 9), int)
        Y = np.empty((0, 3), int)

        # waiting times at stops
        X_stop_top = np.empty((0, 2 + L), int)
        Y_stop_top = np.empty((0, 2), int)

        X_stop = np.empty((0, 7), int)
        Y_stop = np.empty((0, 2), int)

        prev = -1
        m = 0

        for i in range(0, x_travels2.shape[0]):
            if x_date2[i] != prev:
                m = 0
                prev = x_date2[i]
            else:
                m += 1

            # top chunk
            if m < M:
                x = np.concatenate((x_travels2[i, j - L:j], [x_start2[i]], [x_weekday_not2[i]]))
                y = np.array((x_travels2[i, j], x_travels2[i, j + 1], x_travels2[i, j + 2]))
                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X_top = np.append(X_top, np.array([x]), axis=0)
                    Y_top = np.append(Y_top, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass

                x = np.concatenate((x_stops1[i, j - L:j], [x_start2[i]], [x_weekday_not2[i]]))
                y = np.array((x_stops1[i, j], x_stops1[i, j + 1]))
                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X_stop_top = np.append(X_stop_top, np.array([x]), axis=0)
                    Y_stop_top = np.append(Y_stop_top, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass
            # major chunk
            else:
                x = np.concatenate(
                    (x_travels2[i, j - L:j], x_travels2[i - M:i, j:j + 3].ravel(), [x_start2[i]], [x_weekday_not2[i]]))
                y = np.array((x_travels2[i, j], x_travels2[i, j + 1], x_travels2[i, j + 2]))

                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X = np.append(X, np.array([x]), axis=0)
                    Y = np.append(Y, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass

                x = np.concatenate(
                    (x_stops1[i, j - L:j], x_stops1[i - M:i, j:j + 2].ravel(), [x_start2[i]], [x_weekday_not2[i]]))
                y = np.array((x_stops1[i, j], x_stops1[i, j + 1]))

                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X_stop = np.append(X_stop, np.array([x]), axis=0)
                    Y_stop = np.append(Y_stop, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass

        print '--------------------------------------------'
        print 'test models at %dth stop:' % j

        with open('model_travel_top-' + str(j) + '.pkl', 'rb') as fid:
            model_travel_top = cPickle.load(fid)
        MAE_travel_top.append(mean_absolute_error(Y_top, model_travel_top.predict(X_top[:, 0:-1])))

        with open('model_travel_main-' + str(j) + '.pkl', 'rb') as fid:
            model_travel_main = cPickle.load(fid)
        MAE_travel_main.append(mean_absolute_error(Y, model_travel_main.predict(X[:, 0:-1])))

        with open('model_stops_main-' + str(j) + '.pkl', 'rb') as fid:
            model_stops_main = cPickle.load(fid)
        MAE_stops_main.append(mean_absolute_error(Y_stop, model_stops_main.predict(X_stop[:, 0:-1])))

        with open('model_stops_top-' + str(j) + '.pkl', 'rb') as fid:
            model_stops_top = cPickle.load(fid)
        MAE_stop_top.append(mean_absolute_error(Y_stop_top, model_stops_top.predict(X_stop_top[:, 0:-1])))

    return MAE_travel_left, MAE_stops_left, np.mean(MAE_travel_top), \
           np.mean(MAE_stop_top), np.mean(MAE_stop_top), np.mean(MAE_travel_main)
