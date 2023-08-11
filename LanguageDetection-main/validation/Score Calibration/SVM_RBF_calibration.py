# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:16:40 2023

@author: Guido
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
import models_llr as m
import numpy as np
from prettytable import PrettyTable 
import evaluation as ev

train_data = os.path.join(main_dir, "Train.txt")
D, L = f.loadData(train_data)

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
results.field_names = ["Pi Calibration","minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "min Cprim", "actDCF (pi = 0.1)", "actDCF (pi = 0.5)", "Cprim"]

     
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
    scores_pool = np.hstack((scores_pool,m.SVM_RBF(DTR, LTR, DTE, C, K, gamma)))

#calibrate scores using single split approach#
#reshuffle scores and relative labels
p = np.random.permutation(scores_pool.shape[0])
scores_pool = scores_pool[p]
labels_pool = labels_pool[p]

#split calibration set in training set (80%) and validation set (20%)
C_DTR = f.vrow(scores_pool[:int((scores_pool.shape[0]*80)/100)])
C_LTR = labels_pool[:int((scores_pool.shape[0]*80)/100)]
C_DTE = f.vrow(scores_pool[int((scores_pool.shape[0]*80)/100):])
C_LTE = labels_pool[int((scores_pool.shape[0]*80)/100):]

#train a weigthed Linear LogReg with lambda set to 0 (unregularized)
prior_list = [0.1, 0.2, 0.5] # calibrate for different priors
out = "Score calibration - SVM RBF"
for prior in prior_list:
    calibrated_scores = m.logistic_regression(C_DTR, C_LTR, 0, C_DTE, C_LTE, prior, cal = True)
    #compute minDCF and actDCF on calibrated vlaidation set
    plt = ev.bayes_error_plot(calibrated_scores, C_LTE, "")
    plt.ylim([0, 1])
    plt.xlim([-3, 3])
    plt.savefig(f"../BayesPlots/BayesPlot_Best_SVM_RBF_calibrated_{prior}.png", dpi = 200)
    plt.show()
    #compute minDCF and actualDCF for the working points on calibration validation set  
    minDCF = np.zeros(2)
    actDCF = np.zeros(2)
    for i, pi in enumerate(pi_list):
        actDCF[i] = ev.compute_act_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
        minDCF[i] = ev.compute_min_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
    Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
    act_Cprim = np.round((actDCF[0] + actDCF[1])/ 2 , 3)
    results.add_row([prior,np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim, np.round(actDCF[0],3), np.round(actDCF[1],3), act_Cprim ])
    
with open('SVM_RBF_Calibration_Results.txt', 'w') as file:
    file.write(out+results.get_string()+"\nLatex version\n"+results.get_latex_string())
print("Done")



    
