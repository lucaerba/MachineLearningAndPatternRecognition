#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:51:45 2023
This script output a plot comparing the best configuration fro SVM Poly with and without znorm
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
PCA = [None] #number of dimension to keep in PCA

np.random.seed(0)
indexes = np.random.permutation(D.shape[1])  

C_list = np.logspace(-5,1,7).tolist()
K = 10 
d = 2
c = 1

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

#set graph
fig, ax = plt.subplots() 
ax.set_xscale('log')
ax.set(xlabel='C', ylabel='Cprim', title=f'SVM K={K} - Polynomial Kernel (d={d},c={c}) No PCA')
plt.grid(True)
plt.xticks(C_list)

#set table
results = PrettyTable()
results.align = "c"
results.field_names = ["ZNorm","K", "C", "Kernel", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]
 
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
            
            #znorm
            if norm:
                DTR, DTE = f.apply_Z_Norm(DTR,DTE)
            
            #pool test scores and test labels in order to compute the minDCF on complete pool set
            labels_pool = np.hstack((labels_pool,LTE))
            scores_pool = np.hstack((scores_pool,models_llr.SVM_Poly(DTR, LTR, DTE, C, K, d, c)))
         
        #compute minDCF for the current SVM with/without PCA for the 2 working points  
        minDCF = np.zeros(2)
        for i, pi in enumerate(pi_list):
            minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
        #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
        Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
        Cprim_list = np.hstack((Cprim_list,Cprim))
        # add current result to table
        results.add_row([norm, K, C, f"Poly(d={d},c={c})", np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim])
        print(f"\t...computed C={C}, Z-norm={norm}") #feedback print
        
    #plot the graph
    label = "Raw features" if norm else "Z-Norm"
    if norm:
        ax.plot(C_list, Cprim_list, label = label, color ="red")
    else:
        ax.plot(C_list, Cprim_list, label = label, color ="green")
    print(f"\tCprim values for norm = {norm}: {Cprim_list}") #feedback print         

print('Completed SVM Poly with znorm comparison###') #feedback print
plt.legend()
fig.savefig("SVM_Poly_Z_Norm_results.png", dpi=200)
print("Plot saved!")
plt.show()

    # print and save as txt the results table for each K,c,d combination
print(results)
    # data = results.get_string()
    # with open(f'Results/minDCF_SVM_Poly_results/SVM_Poly_K{K_num}d{d}c{c}_ResultsTable.txt', 'w') as file:
    #     file.write(data)



    
