#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:51:45 2023
This script produces the bayes plot for the best RBF SVM configuration
@author: guido
"""

import sys
sys.path.append("../")

import Functions as f
import models_llr
import numpy as np
import evaluation as ev
from prettytable import PrettyTable

D, L = f.loadData("../Train.txt")

# number of folds for K-fold
folds = 5 # can't use K for K-fold because in SVM K is already used
N = int(D.shape[1]/folds)
PCA = 5 #number of dimension to keep in PCA

np.random.seed(0)
indexes = np.random.permutation(D.shape[1])  

#best RBF SVM configuration in validation phase
C = 10 
K = 0.01 
gamma = 0.1 

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

results = PrettyTable()
results.align = "c"
results.field_names = ["minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "min Cprim", "actDCF (pi = 0.1)", "actDCF (pi = 0.5)", "Cprim"]


        
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
    if PCA != None:
        DTR,P = f.PCA(DTR,PCA) # fit PCA to training set
    LTR = L[idxTrain]
    if PCA != None:
        DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
    else:
        DTE = D[:,idxTest]
    LTE = L[idxTest]
    
    #pool test scores and test labels in order to compute the minDCF on complete pool set
    labels_pool = np.hstack((labels_pool,LTE))
    scores_pool = np.hstack((scores_pool,models_llr.SVM_RBF(DTR, LTR, DTE, C, K, gamma)))

plt = ev.bayes_error_plot(scores_pool, labels_pool, "SVM RBF Bayes plot")
plt.ylim([0, 0.4])
plt.xlim([-3, 3])
plt.savefig("BayesPlots/BayesPlot_Best_SVM_RBF.png", dpi = 200)
plt.show()
#compute minDCF and actualDCF for the working points  
minDCF = np.zeros(2)
actDCF = np.zeros(2)
for i, pi in enumerate(pi_list):
    actDCF[i] = ev.compute_act_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
    minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
act_Cprim = np.round((actDCF[0] + actDCF[1])/ 2 , 3)
results.add_row([np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim, np.round(actDCF[0],3), np.round(actDCF[1],3), act_Cprim ])

# print and save as txt the final results table
print(results)
with open('SVM_RBF_actDCF.txt', 'w') as file:
    file.write(results.get_string())


    
