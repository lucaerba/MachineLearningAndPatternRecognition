#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:25:15 2023

@author: guido
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 19:50:50 2023

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

D, L = f.loadData("../Train.txt")

#K-fold cross validation
K = 5
N = int(D.shape[1]/K)
PCA = [5, 4, 3, 2, None] #number of dimension to keep in PCA

#### "Main code" starts here ####
np.random.seed(0)
indexes = np.random.permutation(D.shape[1])    

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]


# list of prior for training weigthed logreg
prior_pi_list = ["Dataset", "Fold", 0.1, 0.5, 0.2]
# compute training dataset prior
d_prior = np.sum(L) / L.shape[0]

# set lambda for log reg
l_list = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10 , 10 ** 2, 10 ** 3 ]
#l_list = [10 ** -5] # for debug

results = PrettyTable()
results.align = "c"
results.field_names = ["PCA", "Prior pi", "lambda", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]

# for each PCA...
for PCA_m in PCA:
    
   
    # for each PCA dim we evaluate a weighted logreg given a pi_prior and different lambda
    for pi_prior in prior_pi_list:
        
        Cprim_list = np.array([]) 
        # we regularize the weigthed logreg with different values of lambda 
        for l in l_list:
            
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
                if PCA_m != None:
                    DTR,P = f.PCA(DTR,PCA_m) # fit PCA to training set
                LTR = L[idxTrain]
                if PCA_m != None:
                    DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
                else:
                    DTE = D[:,idxTest]
                LTE = L[idxTest]
                
                if (pi_prior == "Dataset"):
                    # use dataset prior to train weighted logreg
                    prior = d_prior
                elif(pi_prior == "Fold"):
                    # compute and use current fold prior to train weighted logreg
                    fold_prior = np.sum(LTR) / LTR.shape[0]
                    prior = fold_prior
                else:
                    # use remaining prior from list
                    prior = pi_prior
                
                #pool test scores and test labels in order to compute the minDCF on complete pool set
                labels_pool = np.hstack((labels_pool,LTE))
                scores_pool = np.hstack((scores_pool,models_llr.logistic_regression(DTR, LTR, l, DTE, LTE, prior, cal= True)))
            
            #feedback print
            print(f"Computed PCA={PCA_m}, Prior={pi_prior}, Lambda={l} ...")
            
            #compute minDCF for the current logreg(prior pi, labda) with/without PCA for the 2 working points 
            minDCF = np.zeros(2)
            for i, pi in enumerate(pi_list):
                minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
            #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
            Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
            Cprim_list = np.hstack((Cprim_list,Cprim))
            # add current result to table
            results.add_row([PCA_m, pi_prior, l, np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim])
            
        #feedback print
        print(f"Cprim values for PCA={PCA_m}, Prior={pi_prior}: {Cprim_list}")
        # once computed Cprim for PCA_m, prior and different lambdas plot the graph
        fig,ax = plt.subplots()
        ax.plot(l_list,Cprim_list)
        ax.set_xscale('log')
        ax.set(xlabel='Lambda', ylabel='Cprim', title=f'PCA={PCA_m}, Prior={pi_prior}')
        plt.xticks(l_list)
        #plt.yticks(Cprim_list)
        plt.grid(True)
        fig.savefig(f"Results/minDCF_LogReg_results/PCA{PCA_m}_Prior{pi_prior}.png", dpi=200)
        plt.show()
                
#print and save as txt the results table        
print(results)
data = results.get_string()

with open('Results/minDCF_LogReg_results/LogReg_ResultsTable.txt', 'w') as file:
    file.write(data)
