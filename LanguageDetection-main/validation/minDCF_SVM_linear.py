#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:51:45 2023

@author: guido
"""

import sys
sys.path.append("../")

import Functions as f
import models_llr
import numpy as np
import evaluation as ev
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import datetime


D, L = f.loadData("../Train.txt")

# number of folds for K-fold
folds = 5 # can't use K for K-fold because in SVM K is already used
N = int(D.shape[1]/folds)
PCA = [5, 4, None] #number of dimension to keep in PCA

np.random.seed(0)
indexes = np.random.permutation(D.shape[1])  

C_list = np.logspace(-5,1,7).tolist() # from 10^-4 to 10
K_list = np.logspace(-2,1,4).tolist() # from 10^-2 to 10

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

results = PrettyTable()
results.align = "c"
results.field_names = ["K", "C", "PCA", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]


# produce a graph for each K, on x plot different C used for training on Y plot relative Cprim obtained
for K_num, K in enumerate(K_list):
    st = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"### {st}: Starting SVM linear with K = {K}") #feedback print
    
    #set graph
    fig, ax = plt.subplots() 
    ax.set_xscale('log')
    ax.set(xlabel='C', ylabel='Cprim', title=f'Linear SVM K={K}')
    plt.grid(True)
    plt.xticks(C_list)
    
    for PCA_m in PCA:
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
                if PCA_m != None:
                    DTR,P = f.PCA(DTR,PCA_m) # fit PCA to training set
                LTR = L[idxTrain]
                if PCA_m != None:
                    DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
                else:
                    DTE = D[:,idxTest]
                LTE = L[idxTest]
                
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
            # add current result to table
            results.add_row([K, C, PCA_m, np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim])
            print(f"\t...computed C={C}, PCA={PCA_m}") #feedback print
        #plot the graph
        ax.plot(C_list, Cprim_list, label =f'PCA-{PCA_m}')
        print(f"\tCprim values for PCA={PCA_m}: {Cprim_list}") #feedback print         
    
    print(f'Completed SVM linear with K = {K} ###') #feedback print
    plt.legend()
    fig.savefig(f"Results/minDCF_SVM_Linear_results/SVM_linear_{K_num}.png", dpi=200)
    plt.show()
    
# print and save as txt the final results table
print(results)
data = results.get_string()

with open('Results/minDCF_SVM_Linear_results/SVM_Linear_ResultsTable.txt', 'w') as file:
    file.write(data)

    