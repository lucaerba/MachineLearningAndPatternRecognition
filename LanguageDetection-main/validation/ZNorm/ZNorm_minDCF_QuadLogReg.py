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

### QuadLogReg best results with raw features obtained with: Prior pi = 0.5, No PCA, Lambda = 0.1  
### Cprim was 0.231
# K-fold cross validation
K = 5
N = int(D.shape[1]/K)
PCA_m = None  

#### "Main code" starts here ####
np.random.seed(0)
indexes = np.random.permutation(D.shape[1])

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

# set quadlogreg prior
prior = 0.5
# set lambda for quad log reg
l = 0.1

def vec_xxT(x):
    x = f.vcol(x)  # take it as a column vector
    return np.dot(x, x.T).reshape(x.size ** 2)  # build vec(xxT)


results = PrettyTable()
results.align = "c"
results.field_names = ["Prior pi", "PCA", "Lambda", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]

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
    DTE = D[:, idxTest]
    LTE = L[idxTest]
    
    #Z-Norm
    DTR, DTE = f.apply_Z_Norm(DTR,DTE)
    
    if PCA_m != None:
        DTR, P = f.PCA(DTR, PCA_m)  # fit PCA to training set
        # transform test samples according to P from PCA on dataset
        DTE = np.dot(P.T, D[:, idxTest])

    # features expansion on training dataset, after PCA
    DTR_XXT = np.apply_along_axis(vec_xxT, 0, DTR)
    DTR = np.vstack([DTR_XXT, DTR])

    # feature expansion on eval data, after PCA
    DTE_XXT = np.apply_along_axis(vec_xxT, 0, DTE)
    DTE = np.vstack([DTE_XXT, DTE])

    # pool test scores and test labels in order to compute the minDCF on complete pool set
    labels_pool = np.hstack((labels_pool, LTE))
    scores_pool = np.hstack((scores_pool, models_llr.logistic_regression(DTR, LTR, l, DTE, LTE, prior, cal=True)))

# compute minDCF for the current logreg(prior pi, labda) with/without PCA for the 2 working points
minDCF = np.zeros(2)
for i, pi in enumerate(pi_list):
    minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
# compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
Cprim = np.round((minDCF[0] + minDCF[1]) / 2, 3)

# add current result to table
results.add_row([prior, PCA_m, l, np.round(minDCF[0], 3), np.round(minDCF[1], 3), Cprim])

# print and save as txt the results table
out_string = "*** Z-Norm Applied ***\n" + results.get_string() 
print(out_string)
# data = results.get_string()

with open('QuadLogReg_Z_Norm_results.txt', 'w') as file:
     file.write(out_string)
