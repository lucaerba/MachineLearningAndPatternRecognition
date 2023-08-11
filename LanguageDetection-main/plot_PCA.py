#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:57:09 2023
Script that produces a scatter plot of training dataset features after PCA2 applied
@author: guido
"""

import numpy 
import matplotlib.pyplot as plt

def vcol(m):
    """Reshape as a column vector input m"""
    return m.reshape((m.size,1))

def loadData(filename):
    """
    Load dataset samples and relative labels from file. Returns D data-matrix where each column is a sample and L containing labels.
    The i-th sample of D is D[:,i] and the relative label is L[i].
    """
    
    D = []
    L = []
    with open(filename, 'r') as f:
        for line in f:
            sample = numpy.array([float(i) for i in line.split(",")][0:6]).reshape(6,1) # 6 features 
            label = line.split(",")[6].rstrip()
            D.append(sample)
            L.append(label)
        D = numpy.hstack(D)
        L = numpy.array(L, dtype=numpy.int32)
        return D,L

def plot_scatter(D, L, title):
    
    # make data that will be plotted
    ita = (L == 1)
    no_ita = (L == 0)
    
    data_ita = D[:, ita]
    data_no_ita = D[:, no_ita]

    plt.figure()
    plt.scatter(data_no_ita[0, :], data_no_ita[1, :], label='Not Italian', s=16, alpha = 0.5)
    plt.scatter(data_ita[0, :], data_ita[1, :], label='Italian',s=16, alpha = 0.5)

    
    #plt.title(title)
    plt.legend()
    plt.savefig('PCA_2.png', dpi = 200)
    plt.show()
    
    
def PCA(D):
    N = D.shape[1]
    mu = vcol(D.mean(1)) # compute mean by column of the dataset for each dimension, note that mu is a row vector of shape (4,)
    DC = D - mu # center data
    C = numpy.dot(DC, DC.T)/N # compute the covariance matrix of centered data
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:2] # 2 dim PCA
    DP = numpy.dot(P.T, D)
    return DP

if __name__ == "__main__":
    D,L = loadData("Train.txt")
    D_PCA = PCA(D)
    plot_scatter(D_PCA, L, "PCA")
    
    
    
            
