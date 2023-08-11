#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:58:33 2023
Script that plot features of training dataset in histograms
@author: guido
"""

import Functions as f
import matplotlib.pyplot as plt

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    l = {
        0: 'RAW Feature 0',
        1: 'RAW Feature 1',
        2: 'RAW Feature 2',
        3: 'RAW Feature 3',
        4: 'RAW Feature 4',
        5: 'RAW Feature 5',
        }

    for dIdx in range(6):
        plt.figure()
        #plt.xlabel(l[dIdx])
        plt.hist(D0[dIdx, :], bins = 50, density = True, stacked = True, alpha = 0.6, color = "tomato", edgecolor = "black", label = 'Not Italian')
        plt.hist(D1[dIdx, :], bins = 50, density = True, stacked = True, alpha = 0.7, color="mediumslateblue", edgecolor = "black", label = 'Italian')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.png' % dIdx, dpi = 200)
    plt.show()

D,L = f.loadData("Train.txt")
plot_hist(D,L)
