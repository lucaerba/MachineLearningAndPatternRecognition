#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:51:45 2023
This script computes a graph containing values of Cprim for 2 cases with SVM Linear:
    Best configuration obtained without z-norm and the same configuration with ZNorm
    best conf w/out Znorm was K = 0.01, No PCA -> Cprim was 0.74
@author: guido
"""

#import paths
import sys
import os
current = os.path.dirname(os.path.realpath(__file__)) # main_dir/validation/ZNorm
parent_dir = os.path.dirname(current)                 # main_dir/validation
main_dir= os.path.dirname(parent_dir)                 # main_dir
sys.path.append(main_dir)
sys.path.append(parent_dir)

import Functions as f
import models_llr
import numpy as np
from prettytable import PrettyTable 
import evaluation as ev
import matplotlib.pyplot as plt

train_data = os.path.join(main_dir, "Train.txt")
D, L = f.loadData(train_data)

# number of folds for K-fold
folds = 5 # can't use K for K-fold because in SVM K is already used
N = int(D.shape[1]/folds)
PCA = None #number of dimension to keep in PCA

np.random.seed(0)
indexes = np.random.permutation(D.shape[1])  

C_list = np.logspace(-5,1,7).tolist() # from 10^-4 to 10
K = 0.01

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

results = PrettyTable()
results.align = "c"
results.field_names = ["K", "C", "PCA", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]


#set graph
fig, ax = plt.subplots() 
ax.set_xscale('log')
ax.set(xlabel='C', ylabel='Cprim', title=f'Linear SVM with K={K}, No PCA')
plt.grid(True)
plt.xticks(C_list)
  
for norm in [False, True]:   
    Cprim_list = np.array([])
    
    for C in C_list:
        #for each C compute minDCF after K-fold
        scores_pool = np.array([])
        labels_pool = np.array([])
        
        for i in range(folds):
        
            idxTest = indexes[i*N:(i+1)*N]
        
            if i > 0:
                idxTrainLeft = indexes[0:i*N]
            elif (i+1) < folds:
                idxTrainRight = indexes[(i+1)*N:]
        
            if i == 0:
                idxTrain = idxTrainRight
            elif i == folds-1:
                idxTrain = idxTrainLeft
            else:
                idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR = L[idxTrain]
            DTE = D[:,idxTest]
            LTE = L[idxTest]
            
            #ZNorm
            if(norm):
                DTR, DTE = f.apply_Z_Norm(DTR,DTE)
            
            if PCA != None:
                DTR,P = f.PCA(DTR,PCA) # fit PCA to training set
                DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
            
            #pool test scores and test labels in order to compute the minDCF on complete pool set
            labels_pool = np.hstack((labels_pool,LTE))
            scores_pool = np.hstack((scores_pool,models_llr.SVM_linear(DTR, LTR, DTE, C, K)))
         
        #compute minDCF for the current SVM with/without PCA for the 2 working points  
        minDCF = np.zeros(2)
        for i, pi in enumerate(pi_list):
            minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
        #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
        Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
        Cprim_list = np.hstack((Cprim_list,Cprim))
        
    #plot the graph
    label = "Z-Norm" if norm else "Raw features"
    if(norm):
        ax.plot(C_list, Cprim_list, label = label, color = "red")
    else:
        ax.plot(C_list, Cprim_list, label = label, color = "green")
                     
    
plt.legend()
fig.savefig("SVM_Linear_Z_Norm_results.png", dpi=200)
plt.show()




    