#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:51:45 2023
This script performs a grid search approach with test data and SVM RBF
@author: guido
"""
import sys
sys.path.append("../")

import Functions as f
from validation import models_llr
import numpy as np
import evaluation as ev
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import datetime

cal_prior = [(0.1,"01"), (0.2,"02"), (0.5,"05")] # score calibration 

PCA = [None, 5, 4 ] #number of dimension to keep in PCA

C_list = np.logspace(-5,1,7).tolist() 
K_list = np.logspace(-2,1,4).tolist() 
gamma_list = np.logspace(-3,0,4).tolist() 

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]



for prior_cal, prior_str in cal_prior:
    
    st = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"### {st}: Starting calibration for {prior_cal}") #feedback print
    results = PrettyTable()
    results.align = "c"
    results.field_names = ["calib. prior", "K", "C", "PCA", "Kernel", "min Cprim"]
    
    # produce a graph for each K,c,d: on x plot different C used for training, on Y plot relative Cprim obtained
    for K_num, K in enumerate(K_list):
        
        for PCA_m in PCA:
        
            st = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"### {st}: Starting SVM RBF with K = {K}, PCA = {PCA_m}") #feedback print
    
            #set graph
            fig, ax = plt.subplots() 
            ax.set_xscale('log')
            ax.set(xlabel='C', ylabel='Cprim', title=f'SVM K={K} - RBF Kernel')
            plt.grid(True)
            plt.xticks(C_list)
        
            for gamma in gamma_list:    
                Cprim_list = np.array([])
                
                for C in C_list:
                    #for each C compute minDCF
                    
                    # load training and test data from scratch
                    DTR, LTR = f.loadData('../Train.txt')
                    DTE, LTE = f.loadData('../Test.txt')
                    
                    #shuffle them
                    np.random.seed(0)
                    index_TR = np.random.permutation(DTR.shape[1])
                    DTR = DTR[:,index_TR]
                    LTR = LTR[index_TR]
                    index_TE = np.random.permutation(DTE.shape[1])
                    DTE = DTE[:,index_TE]
                    LTE = LTE[index_TE]
                    
                    if PCA_m != None:
                        DTR,P = f.PCA(DTR,PCA_m) # fit PCA to training set
                        DTE = np.dot(P.T,DTE) # transform test samples according to P from PCA on dataset
                
                    scores = models_llr.SVM_RBF(DTR, LTR, DTE, C, K, gamma)
                    
                    #calibrate scores using single split approach#
                    #reshuffle scores and relative labels
                    p = np.random.permutation(scores.shape[0])
                    scores = scores[p]
                    LTE = LTE[p]
    
                    #split calibration set in training set (80%) and validation set (20%)
                    C_DTR = f.vrow(scores[:int((scores.shape[0]*80)/100)])
                    C_LTR = LTE[:int((scores.shape[0]*80)/100)]
                    C_DTE = f.vrow(scores[int((scores.shape[0]*80)/100):])
                    C_LTE = LTE[int((scores.shape[0]*80)/100):]
    
                    #train a weigthed Linear LogReg with lambda set to 0 (unregularized)
                    prior = prior_cal
                    calibrated_scores = models_llr.logistic_regression(C_DTR, C_LTR, 0, C_DTE, C_LTE, prior, cal = True)
                 
                    #compute minDCF for the current SVM with/without PCA for the 2 working points  
                    minDCF = np.zeros(2)
                    for i, pi in enumerate(pi_list):
                        minDCF[i] = ev.compute_min_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
                    #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
                    min_Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
                    Cprim_list = np.hstack((Cprim_list,min_Cprim))
                    # add current result to table
                    results.add_row([prior_cal,K, C, PCA_m, f"RBF(gamma = {gamma}) ", min_Cprim])
                    print(f"\t...computed C={C}, γ={gamma}") #feedback print
                        
                #plot the graph
                ax.plot(C_list, Cprim_list, label =f'log(gamma)={np.log10(gamma)}')
                print(f"\tCprim values for γ={gamma}: {Cprim_list}") #feedback print         
        
            print(f'Completed SVM RBF with K = {K} and PCA = {PCA_m} ###') #feedback print
            plt.legend()
            fig.savefig(f"Results/SVM_RBF_results/CAL{prior_str}_SVM_RBF_K{K_num}_PCA{PCA_m}.png", dpi=200)
            plt.show()
    
    # print and save as txt the final results table for each calibration prior pi
    print(results)
    data = results.get_string()
    
    #for each cal prior a result table
    with open(f'Results/SVM_RBF_results/CAL{prior_str}_SVM_RBF_Eval_Results.txt', 'w') as file:
        file.write(results.get_string())

    
