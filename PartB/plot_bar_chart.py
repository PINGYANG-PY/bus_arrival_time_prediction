#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_bar_chart(means, stds):
    means=np.array(means)
    stds=np.array(stds)
    width = 0.08  # the width of the bars
    rects=[]
    fig, ax = plt.subplots(figsize=(15,5))
    for i in range(0, means.shape[0]):

        N = means.shape[1]
        # mean = [20, 35, 30, 35, 27]
        # std = [2, 3, 4, 1, 2]
        #
        ind = np.arange(N)  # the x locations for the groups
        rects.append(ax.bar(ind+i*width, means[i,:], width, yerr=stds[i,:]))

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('absolute test error of different models for waiting times at stop2 ... stop10')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('KNN', 'ran-forest', 'linearR', 'naiveMean'))
    plt.ylim((5,35))
    plt.show()



