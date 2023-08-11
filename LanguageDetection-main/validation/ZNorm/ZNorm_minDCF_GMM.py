# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:44:46 2023
Apply ZNorm on best GMM conf
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
import models_llr
import numpy as np
import evaluation as ev
from prettytable import PrettyTable

train_data = os.path.join(main_dir, "Train.txt")
D, L = f.loadData(train_data)

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

# K-fold settings
K = 5 # can't use K for K-fold because in SVM K is already used
N = int(D.shape[1]/K)
np.random.seed(0)
indexes = np.random.permutation(D.shape[1]) 

#PCA dimensions to use
PCA = [None]

# set GMM-LBG doublings and GMM version to use for both target and non-target class, number of components for GMM model will be 2^(doub_target/nonTarget)
version_target = "full"
doub_target_list = [0]
version_nonTarget = "tied"
doub_nonTarget_list = [6]

final_output = f"Z-Norm on best GMM configuration\nTarget Class GMM: {version_target}, Non-Target Class GMM: {version_nonTarget}\n"

for doub_target in doub_target_list:
    for doub_nonTarget in doub_nonTarget_list:
        # set results table 
        results = PrettyTable()
        results.align = "c"
        results.field_names = ["PCA", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]
        
        output_string = f"Target class with {2 ** doub_target} components - GMM{version_target}\nNon-Target class with {2 ** doub_nonTarget} components - GMM{version_nonTarget}\n"
        
        for PCA_m in PCA:
            
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
                
                # Z-Norm
                DTR, DTE = f.apply_Z_Norm(DTR,DTE)
            
                if PCA_m != None:
                    DTR,P = f.PCA(DTR,PCA_m) # fit PCA to training set
                    DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset                    
                
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
                    
                #pool test scores and test labels in order to compute the minDCF on complete pool set
                labels_pool = np.hstack((labels_pool,LTE))
                scores_pool = np.hstack((scores_pool,scores))
             
            #compute minDCF for the current SVM with/without PCA for the 2 working points  
            minDCF = np.zeros(2)
            for i, pi in enumerate(pi_list):
                minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
            #compute Cprim (the 2 working points average minDCF) 
            Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
            # add current result to table
            results.add_row([PCA_m, np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim])
           
        final_output += output_string + results.get_latex_string() + "\n" # save in latex table form
        output_string += results.get_string()    
        print(output_string)
        

with open('GMM_Z_NORM_results.txt', 'w') as file:
    file.write(final_output)