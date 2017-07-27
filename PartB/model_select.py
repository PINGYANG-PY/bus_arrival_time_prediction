#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  model is selected using cross-validation on training set
# 1. select and align the timestamped variables (stop times, travel times) in a window to a sample vector.
#           this vector is used as indepedant variables for  prediction
# 2. random forest regression suggests x_vehicle, x_weekday_not are of little relevance. Both are removed.


# Variable Assembling
# To predict the arrival time of the next three stops,
# we predict when the bus leaves the previous stop (j-1) and starts to travele toward the next stop (j-1->j),
# we need to predict 3 travel segments and  2 stops
#
# TRAVE|     |____|     | ////|     |/////|     |/////|     |____|     |____|     |____|     |____|     |____|     |____|     |____
# TIME |     |____|     | ////|     |/////|     |/////|     |____|     |____|     |____|     |____|     |____|     |____|     |____
#  ____|     |////|     | x_j |     |x_j+1|     |x_j+2|     |____|     |____|     |____|     |____|     |____|     |____|     |____
#
#  ____|     |____|     |_____|     |_____|     |_____|     |____|     |____|     |____|     |____|     |____|     |____|     |____
#  ____|     |____|     |_____|/////|_____|/////|_____|     |____|     |____|     |____|     |____|     |____|     |____|     |____
# STOP |     |____|     |_____|/////|_____|/////|_____|     |____|     |____|     |____|     |____|     |____|     |____|     |____
# TIME_|     |____|/////|_____| y_j |_____|y_j+1|     |     |____|     |____|     |____|     |____|     |____|     |____|     |____
#  ____|     |____|     |_____|     |_____|     |_____|     |____|     |____|     |____|     |____|     |____|     |____|     |____
#                       |-----------------------------|
#                       predict: 3 segments+2 stops   |
#                             |           |           |
#                             |           |           |
#                            1st next    2nd next   3rd nextstops
#
#  For the traveling time prediction, I will use
#  the travelling time for the most recent L spans of the current bus, and the traveling times for the next 3 spans of the previous M buses
#  a regression model needs to be built for each prediction
#




import numpy as np
from find_best_model import best_model, candidate_models, candidate_models_0


def model_select(x_weekday_not1, x_start1, x_stops1, x_travels1, x_date1, tolerate=False):
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

    # KNN is used to predict the traveling/waiting times for stops 1 to stop L, travel spans 1 to span L,
    # select the number of neighbours and
    # compare the model performance with naive historical mean
    (name, best_model_start_travel, _, _) = best_model(candidate_models_0(), X_left, Y_left)
    print 'travel-start model is %s:' % name

    (name, best_model_start_stop, _, _) = best_model(candidate_models_0(), X_stop_left, Y_stop_left)
    print 'stop-start model is %s:' % name

    # now, select the models for  stops/spans from L to terminal
    best_models_top_travel = []
    best_models_main_travel = []
    best_models_top_stop = []
    best_models_main_stop = []

    model_errors_top_travel = []
    model_errors_main_travel = []
    model_errors_top_stop = []
    model_errors_main_stop = []

    model_stds_top_travel = []
    model_stds_main_travel = []
    model_stds_top_stop = []
    model_stds_main_stop = []

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
        print 'predication models at %dth stop:' % j

        (name, best_model_tmp, test_errors, error_stds) = best_model(candidate_models(), X_top[:, 0:-1], Y_top)
        print '1. best top-travel model is %s:' % name
        best_models_top_travel.append(best_model_tmp)
        model_errors_top_travel.append(test_errors)
        model_stds_top_travel.append(error_stds)

        (name, best_model_tmp, test_errors, error_stds) = best_model(candidate_models(), X[:, 0:-1], Y)
        print '2. best main-travel model is %s:' % name
        best_models_main_travel.append(best_model_tmp)
        model_errors_main_travel.append(test_errors)
        model_stds_main_travel.append(error_stds)

        (name, best_model_tmp, test_errors, error_stds) = best_model(candidate_models(), X_stop_top[:, 0:-1],
                                                                     Y_stop_top)
        print '3. best top-stop model is %s:' % name
        best_models_top_stop.append(best_model_tmp)
        model_errors_top_stop.append(test_errors)
        model_stds_top_stop.append(error_stds)

        (name, best_model_tmp, test_errors, error_stds) = best_model(candidate_models(), X_stop[:, 0:-1], Y_stop)
        print '4. best main-stop model is %s:' % name
        best_models_main_stop.append(best_model_tmp)
        model_errors_main_stop.append(test_errors)
        model_stds_main_stop.append(error_stds)

    return model_errors_top_travel, model_errors_main_travel, \
           model_errors_top_stop, model_errors_main_stop, \
           model_stds_top_travel, model_stds_main_travel, \
           model_stds_top_stop, model_stds_main_stop
