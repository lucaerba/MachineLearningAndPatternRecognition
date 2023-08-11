# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:44:46 2023

@author: Guido
"""
import sys
sys.path.append('../')
import numpy as np
import Functions as f
import evaluation as ev
from validation import models_llr
from prettytable import PrettyTable
#import datetime


# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]


#PCA dimensions to use
PCA = [None, 5]

# set GMM-LBG doublings and GMM version to use for both target and non-target class, number of components for GMM model will be 2^(doub_target/nonTarget)
version_target = "tied"
doub_target_list = [0]
version_nonTarget = "tied"
doub_nonTarget_list = [2,3,4,5,6]

final_output = f"Target Class GMM: {version_target}, Non-Target Class GMM: {version_nonTarget}\n"

for norm in [False, True]:
    for doub_target in doub_target_list:
        for doub_nonTarget in doub_nonTarget_list:
            # set results table 
            results = PrettyTable()
            results.align = "c"
            results.field_names = ["PCA", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "min Cprim"]
            
            output_string = f"Target class with {2 ** doub_target} components - GMM{version_target}\nNon-Target class with {2 ** doub_nonTarget} components - GMM{version_nonTarget}\n"
            output_string += f"Z-Norm:{norm}"
            for PCA_m in PCA:
                
                DTR, LTR = f.loadData("../Train.txt")
                DTE, LTE = f.loadData("../Test.txt")
                
                if norm:
                    DTR, DTE = f.apply_Z_Norm(DTR,DTE)

                if PCA_m != None:
                    DTR,P = f.PCA(DTR,PCA_m) # fit PCA to training set
                    DTE = np.dot(P.T,DTE) # transform test samples according to P from PCA on dataset
                    
                # get training samples of class 0, train a GMM on them, compute SM for class 0
                DTR0=DTR[:,LTR==0]                                  
                gmm_class0=models_llr.GMM_LBG(DTR0, doub_nonTarget, version_nonTarget)  
                _, SM0=models_llr.logpdf_GMM(DTE,gmm_class0)                    
               
                # same for class 1
                DTR1=DTR[:,LTR==1]                                  
                gmm_class1= models_llr.GMM_LBG(DTR1, doub_target, version_target)
                _, SM1=models_llr.logpdf_GMM(DTE,gmm_class1)
                
                # compute scores
                scores = SM1 - SM0 
                    
                #compute minDCF for the 2 working points  
                minDCF = np.zeros(2)
                for i, pi in enumerate(pi_list):
                    minDCF[i] = ev.compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
                #compute Cprim (the 2 working points average minDCF) 
                min_Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
                # add current result to table
                results.add_row([PCA_m, np.round(minDCF[0],3), np.round(minDCF[1],3), min_Cprim])
               
            final_output += output_string + "\n" + results.get_string() + "\n"
            print(output_string +"\n"+ results.get_string())
            
# save results
with open(f'Results/GMM_results/GMM{version_target}_{version_nonTarget}_evaluation_results.txt', 'w') as file:
    file.write(final_output)