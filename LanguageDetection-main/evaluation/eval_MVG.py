#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:25:15 2023

@author: guido
"""
import sys
sys.path.append("../")
import Functions as f
from validation import models_llr
import numpy as np
from prettytable import PrettyTable 
import evaluation as ev

 

# set PCA, if value is None will not be applied
PCA = [None, 6, 5, 4, 3, 2]


Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

models = [(models_llr.MVG_log , "MVG"), 
          (models_llr.NaiveBayesGaussianClassifier, "Naive Bayes"),
          (models_llr.TiedCovarianceGaussianClassifier, "Tied Covariance")]

txt_output = ""

for model,m_string in models:

    results = PrettyTable()
    results.align = "c"
    results.field_names = ["PCA","minDCF (pi = 0.1)","minDCF (pi = 0.5)", "Cprim"]
    
    
    
    for PCA_m in PCA:    
        
        DTR, LTR = f.loadData("../Train.txt")
        DTE, LTE = f.loadData("../Test.txt")
        
        if PCA_m != None:
            DTR,P = f.PCA(DTR,PCA_m)
            DTE = np.dot(P.T,DTE)
            
        scores = model(DTR, LTR, DTE, LTE)     
        # compute minDCF for current model   
        minDCF = np.zeros(2)
        for i, p in enumerate(pi_list):
            minDCF[i] = ev.compute_min_DCF(scores, LTE, p, Cfn, Cfp)    
        # add minDCF results to table
        results.add_row([PCA_m, np.round(minDCF[0],3), np.round(minDCF[1],3), np.round((minDCF[0] + minDCF[1]) / 2, 3)])
        
    print(f"-{m_string}\n{results}")   
    # concat current model results in output string for the txt results file
    txt_output = txt_output + "\n" + f"-{m_string}\n" + str(results) 

# save output txt file containing results for each model
with open('Results/MVG_results/Eval_MVG_results.txt', 'w') as file:
    file.write(str(txt_output))     


