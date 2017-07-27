#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
import cPickle


# KNN is used for predict the waiting and traveling time for the starting L stops
# random forests with  estimators=40 and max_depth = 2 is select for predict the traveling time for the first M buses in the morning
# Naive historical mean is used to predict the waiting time at all the stops for the first M buses in the morning
# linear regressor is select for predict the traveling time between stops L to terminal and the waiting time at the stops L to terminal
# fitted models are all pickled.

def model_training(x_weekday_not1, x_start1, x_stops1, x_travels1, x_date1):
    M = 2
    L = 1

    # this part is only used for predicting stops 1 to stop L, travel spans 1 to span L, （Note:stop2 in the original record became stop1 now)
    # x_weekday_not is kept for possible future use.
    X_left = np.column_stack((x_start1, x_weekday_not1[:, np.newaxis]))
    Y_left = x_travels1[:, 0:L]

    X_stop_left = np.column_stack((x_start1, x_weekday_not1[:, np.newaxis]))
    Y_stop_left = x_stops1[:, 0:L]

    # for left start part,  drop the missing data
    mask = np.isnan(Y_left).sum(axis=1) > 0
    X_left = X_left[~mask, :]
    Y_left = Y_left[~mask, :]

    mask = np.isnan(Y_stop_left).sum(axis=1) > 0
    X_stop_left = X_stop_left[~mask, :]
    Y_stop_left = Y_stop_left[~mask, :]

    n_neighbors = 150
    knn = KNeighborsRegressor(n_neighbors, weights='uniform')
    model_travel_left = knn.fit(X_left, Y_left)
    model_stops_left = knn.fit(X_stop_left, Y_stop_left)

    # save the classifier
    with open('model_travel_left.pkl', 'wb') as fid:
        cPickle.dump(model_travel_left, fid)
    with open('model_stops_left.pkl', 'wb') as fid:
        cPickle.dump(model_stops_left, fid)

    # now, train the models for  stops/spans from L to terminal
    for j in range(L, x_travels1.shape[1] - 2):
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

        for i in range(0, x_travels1.shape[0]):
            if x_date1[i] != prev:
                m = 0
                prev = x_date1[i]
            else:
                m += 1

            # top chunk
            if m < M:
                x = np.concatenate((x_travels1[i, j - L:j], [x_start1[i]], [x_weekday_not1[i]]))
                y = np.array((x_travels1[i, j], x_travels1[i, j + 1], x_travels1[i, j + 2]))
                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X_top = np.append(X_top, np.array([x]), axis=0)
                    Y_top = np.append(Y_top, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass

                x = np.concatenate((x_stops1[i, j - L:j], [x_start1[i]], [x_weekday_not1[i]]))
                y = np.array((x_stops1[i, j], x_stops1[i, j + 1]))
                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X_stop_top = np.append(X_stop_top, np.array([x]), axis=0)
                    Y_stop_top = np.append(Y_stop_top, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass
            # major chunk
            else:
                x = np.concatenate(
                    (x_travels1[i, j - L:j], x_travels1[i - M:i, j:j + 3].ravel(), [x_start1[i]], [x_weekday_not1[i]]))
                y = np.array((x_travels1[i, j], x_travels1[i, j + 1], x_travels1[i, j + 2]))

                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X = np.append(X, np.array([x]), axis=0)
                    Y = np.append(Y, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass

                x = np.concatenate(
                    (x_stops1[i, j - L:j], x_stops1[i - M:i, j:j + 2].ravel(), [x_start1[i]], [x_weekday_not1[i]]))
                y = np.array((x_stops1[i, j], x_stops1[i, j + 1]))

                if (tolerate == False and ~np.isnan(x).any() and ~np.isnan(y).any()):
                    X_stop = np.append(X_stop, np.array([x]), axis=0)
                    Y_stop = np.append(Y_stop, np.array([y]), axis=0)
                elif (tolerate == True and np.isnan(x).sum() == 1 and ~np.isnan(y).any()):
                    pass

        print '--------------------------------------------'
        print 'train models at %dth stop:' % j

        model_travel_top = RandomForestRegressor(n_estimators=40, max_depth=2, random_state=2)
        model_travel_top.fit(X_top[:, 0:-1], Y_top)
        with open('model_travel_top-' + str(j) + '.pkl', 'wb') as fid:
            cPickle.dump(model_travel_top, fid)

        model_travel_main = LinearRegression()
        model_travel_main.fit(X[:, 0:-1], Y)
        with open('model_travel_main-' + str(j) + '.pkl', 'wb') as fid:
            cPickle.dump(model_travel_main, fid)

        model_stops_main = LinearRegression()
        model_stops_main.fit(X_stop[:, 0:-1], Y_stop)
        with open('model_stops_main-' + str(j) + '.pkl', 'wb') as fid:
            cPickle.dump(model_stops_main, fid)

        model_stops_top = DummyRegressor()
        DummyRegressor.fit(X_stop_top[:, 0:-1], Y_stop_top)
        with open('model_stops_top-' + str(j) + '.pkl', 'wb') as fid:
            cPickle.dump(model_stops_top, fid)
