#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 17:01:35 2023

This script computes most discriminant LDA direction on training dataset and shows an histogram result 

@author: guido
"""

import Functions as f
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

def LDA(D,L,directions = 1):
    "Perform LDA on a dataset D where each column is a sample and each row a feature"
    NC = 2 # number of classes in the dataset
    N = D.shape[1] # number of samples in dataset
    mu = f.vcol((D.mean(1))) 
    samples = [] # will contain the samples of i-th class (0,1,2) in the i-th position
    muC = [] # will contain the mean of i-th class in the i-th position as column vector
    for i in range(NC):
        samples.append(D[: , L==i])  # get samples for each class
        muC.append(f.vcol(D[: , L==i].mean(1))) #compute mean for each class
                   
    #compute SW
    SWC = [] # will contain SW,c for each class where SW,c is the covariance matrix of each class
    for i in range(NC):
        #CC is for Centered Class, the samples of a specific class centered (subtractaction of class mean from all class samples)
        CC = samples[i] - muC[i]
        SWC.append(np.dot(CC,CC.T)/samples[i].shape[1]) #compute SW for the i-th class
    
    s=0 # will contain sum of (SW,c * Nc) where Nc is the number of samples for a class
    for i in range(NC):
        s += SWC[i] * samples[i].shape[1]
        SW = s/N # compute the SW matrix
    #compute SB
    summation=0
    for i in range(NC):
        temp = muC[i] - mu
        summation += np.dot(temp, temp.T) * samples[i].shape[1]
    SB = summation/N
    # solve the Generalized eigenvalue problem
    m = directions 
    s, U = sl.eigh(SB,SW)
    W = U[:, ::-1][:,0:m] # matrix W is the one that makes possible to perform LDA projecting samples in the new space
    return np.dot(W.T, D) # dataset after LDA
    

def plot_hist(D, L, title):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    
    plt.figure()
    plt.hist(D0[0,:], bins = 150, alpha = 0.8, label = 'Not Italian')
    plt.hist(D1[0,:], bins = 150, alpha = 1, label = 'Italian')
    plt.yscale("linear")
    plt.legend()
    
    plt.savefig('LDA.png', dpi = 200)
    #plt.title(title)
    plt.show()
    

if __name__ == "__main__":
    D, L = f.loadData('Train.txt')
    D_LDA = LDA(D,L)
    
    
    #plot_hist(D, L, 'Training dataset')
    plot_hist(D_LDA, L, 'Histogram of dataset features - LDA direction')
    
    
