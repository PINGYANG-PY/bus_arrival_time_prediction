#!/usr/bin/python
# -*- coding: UTF-8 -*-

# best_model(candidate_models, X_train, y_train)
# Returns the best model configured using GridSearchCV.

from __future__ import print_function
from sklearn.model_selection import GridSearchCV
import sys

# only for the left start part, select number of neighbours for KNN
def candidate_models_0():
    candidates = []

    from sklearn.neighbors import KNeighborsRegressor
    knn_tuned_parameters = [{'n_neighbors': [30, 70, 150]}]
    candidates.append(['KNeighborsRegressor', KNeighborsRegressor(weights='uniform'),
                       knn_tuned_parameters])

    from sklearn.dummy import DummyRegressor
    candidates.append(['Naive historical mean', DummyRegressor(),
                       [{}]])

    return candidates


# for the main models, compare KNN, Random forest, linear regressor
def candidate_models():
    candidates = []

    from sklearn.neighbors import KNeighborsRegressor
    knn_tuned_parameters = [{'n_neighbors': [20, 30]}]
    candidates.append(['KNeighborsRegressor', KNeighborsRegressor(weights='uniform'),
                       knn_tuned_parameters])

    from sklearn.ensemble import RandomForestRegressor
    RandomForestRegressor_params = [{'max_depth': [1, 2],
                                     'n_estimators': [40, 60]}]
    candidates.append(['RandomForestRegressor', RandomForestRegressor(),
                       RandomForestRegressor_params])

    from sklearn.linear_model import LinearRegression
    LinearRegression_params = [{}]
    candidates.append(['LinearRegression', LinearRegression(),
                       LinearRegression_params])

    from sklearn.dummy import DummyRegressor
    candidates.append(['Naive historical mean', DummyRegressor(),
                       [{}]])

    return candidates


def best_tune(model_name, model, parameters, X_train, y_train, verbose=False):
    clf = GridSearchCV(model, parameters, n_jobs=-1, cv=5,
                       scoring='neg_mean_absolute_error')

    clf.fit(X_train, y_train)

    # The score (neg_mean_absolute_error) is changed to mean_absolute_error (positive)
    mean_abs_error = -1 * clf.cv_results_['mean_test_score'][clf.best_index_]
    std = clf.cv_results_['std_test_score'][clf.best_index_]
    if verbose:
        print('Best hyperparameters found for %s, the mean absolute test error is:' % model_name)
        print('%0.3f (+/-%0.03f) for %r' % (mean_abs_error, std * 2, clf.best_params_))

    return [model_name + str(clf.best_params_), clf.best_score_,
            clf.best_params_, mean_abs_error, std]


def best_model(candidate_models, X_train, y_train, verbose=False):
    best_score = -sys.maxint - 1
    best_model = None
    tuned_models = []
    if verbose:
        print()
        print('-----------------------------------------------')
        print('Grid searching ... ')
    for model_name, model, parameters in candidate_models:
        tuned_models.append(best_tune(model_name, model, parameters,
                                      X_train, y_train))

    mean_errors = []
    std_errors = []
    for name, score, model, mean_abs_error, std in tuned_models:
        mean_errors.append(mean_abs_error)
        std_errors.append(std)
        if (score > best_score):
            best_score = score
            best_model = [name, model]

    if verbose:
        print('Best model is %s ' % best_model[0])

    return best_model[0], best_model[1], mean_errors, std_errors
