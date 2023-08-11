#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:25:15 2023
This script outputs a graph for a LogReg linear with a specific configuration of its parameters and PCA applied, plotting the result
obtained with and without Z-norm
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
import matplotlib.pyplot as plt
import evaluation as ev

train_data = os.path.join(main_dir, "Train.txt")
D, L = f.loadData(train_data)

# compute training dataset prior
d_prior = np.sum(L) / L.shape[0]

# K-fold cross validation
K = 5
N = int(D.shape[1]/K)
PCA = [4]  

#### "Main code" starts here ####
np.random.seed(0)
indexes = np.random.permutation(D.shape[1])

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

# set logreg prior
prior = d_prior 
# set lambda for log reg
l_list = np.logspace(-5, 3, 9).tolist()

fig,ax = plt.subplots()
for normalize in [False,True]:
        
    # for each PCA...
    for PCA_m in PCA:
              
        Cprim_list = np.array([]) 
        # we regularize the weigthed logreg with different values of lambda 
        for l in l_list:
            
            scores_pool = np.array([])
            labels_pool = np.array([])
            
            for i in range(K):
            
                idxTest = indexes[i*N:(i+1)*N]
            
                if i > 0:
                    idxTrainLeft = indexes[0:i*N]
                elif (i+1) < K:
                    idxTrainRight = indexes[(i+1)*N:]
            
                if i == 0:
                    idxTrain = idxTrainRight
                elif i == K-1:
                    idxTrain = idxTrainLeft
                else:
                    idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
            
                DTR = D[:, idxTrain]
                LTR = L[idxTrain]
                DTE = D[:,idxTest]
                LTE = L[idxTest]
                
                if normalize:
                    # Z-Norm
                    DTR, DTE = f.apply_Z_Norm(DTR,DTE)
                
                if PCA_m != None:
                    DTR,P = f.PCA(DTR,PCA_m) # fit PCA to training set
                    DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
    
                #pool test scores and test labels in order to compute the minDCF on complete pool set
                labels_pool = np.hstack((labels_pool,LTE))
                scores_pool = np.hstack((scores_pool,models_llr.logistic_regression(DTR, LTR, l, DTE, LTE, prior, cal= True)))
            
            
            #compute minDCF for the current logreg(prior pi, labda) with/without PCA for the 2 working points 
            minDCF = np.zeros(2)
            for i, pi in enumerate(pi_list):
                minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
            #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
            Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
            Cprim_list = np.hstack((Cprim_list,Cprim))            
        
        # once computed Cprim for PCA_m, prior and different lambdas plot the graph
        label = "Z-Norm" if normalize  else "Raw features" 
        ax.plot(l_list,Cprim_list, label = label)
        ax.set_xscale('log')
        ax.set(xlabel='Lambda', ylabel='Cprim', title=f'PCA={PCA_m}, Prior=Dataset')
        plt.xticks(l_list)

plt.grid(True)
plt.legend()
fig.savefig("LogReg_Z_Norm_results.png", dpi=200)
plt.show()
                
