#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:25:15 2023

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

train_data = os.path.join(main_dir, "Train.txt")
D, L = f.loadData(train_data)

#K-fold cross validation
K = 5
N = int(D.shape[1]/K)

# set PCA, if value is None will not be applied
PCA = [None, 6, 5, 4, 3, 2]

#### "Main code" starts here ####
np.random.seed(0)
indexes = np.random.permutation(D.shape[1])    

Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

# tied covariance didn't perfom well so we don't test it with znorm
models = [(models_llr.MVG_log , "MVG"), 
          (models_llr.NaiveBayesGaussianClassifier, "Naive Bayes")]


txt_output = "*** Z-Norm is applied ***"

for model,m_string in models:

    results = PrettyTable()
    results.align = "c"
    results.field_names = ["PCA","minDCF (pi = 0.1)","minDCF (pi = 0.5)", "Cprim"]
    
    for PCA_m in PCA:    
       # do k fold for current model and for each PCA dim 
        labels_pool = np.array([])
        scores_pool = np.array([])
        
        
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
            DTE = D[:, idxTest]
            LTE = L[idxTest]
            
            # apply ZNORM
            DTR, DTE = f.apply_Z_Norm(DTR,DTE)
            
            if PCA_m != None:
                DTR,P = f.PCA(DTR,PCA_m) # fit PCA to training set and get P
                DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P 

            #pool test scores and test labels in order to compute the minDCF on complete pool set after
            labels_pool = np.hstack((labels_pool,LTE))
            scores_pool = np.hstack((scores_pool,model(DTR, LTR, DTE, LTE)))
            
        # compute minDCF for current model after k-fold for each pi on pooled scores and labels    
        minDCF = np.zeros(2)
        for i, p in enumerate(pi_list):
            minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, p, Cfn, Cfp)    
        # add minDCF results to table
        results.add_row([PCA_m, np.round(minDCF[0],3), np.round(minDCF[1],3), np.round((minDCF[0] + minDCF[1]) / 2, 3)])
        
    #print(f"-{m_string}\n{results}")   
    # concat current model results in output string for the txt results file
    txt_output = txt_output + "\n" + f"-{m_string}\n" + str(results) 
print(txt_output)

# save output txt file containing results for each model
with open('MVG_Z_Norm_results.txt', 'w') as file:
     file.write(str(txt_output))     


