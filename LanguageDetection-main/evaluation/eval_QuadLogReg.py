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

import matplotlib.pyplot as plt
from prettytable import PrettyTable
import evaluation as ev
import numpy as np
from validation import models_llr
import Functions as f



PCA = [None, 5,4,3]  # number of dimension to keep in PCA

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

D,L = f.loadData("../Train.txt")
# list of prior for training weigthed logreg
prior_pi_list = ["Dataset", "Fold", 0.1, 0.5, 0.2]

# compute training dataset prior
d_prior = np.sum(L) / L.shape[0]

# set lambda for log reg
l_list = [10 ** -5, 10 ** -4, 10 ** -3,
          10 ** -2, 10 ** -1, 1, 10, 10 ** 2, 10 ** 3]
# l_list = [10 ** -5] # for debug


def vec_xxT(x):
    x = f.vcol(x)  # take it as a column vector
    return np.dot(x, x.T).reshape(x.size ** 2)  # build vec(xxT)


results = PrettyTable()
results.align = "c"
results.field_names = ["Prior pi", "PCA", "λ", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "min Cprim"]

# for each pi prior a graph containing the Cprim values for each PCA dimension will be produced
for pi_prior in prior_pi_list:
    
    print(f"Starting prior {pi_prior}...")
    
    #set graph
    fig, ax = plt.subplots() 
    ax.set_xscale('log')
    ax.set(xlabel='λ', ylabel='Cprim', title=f'QuadLogReg Prior={pi_prior}')
    plt.grid(True)
    plt.xticks(l_list)
    
    for PCA_m in PCA:
        
        Cprim_list = np.array([])
        # we regularize with different values of lambda
        for l in l_list:

            DTR, LTR = f.loadData("../Train.txt")
            DTE, LTE = f.loadData("../Test.txt")

            if PCA_m != None:
                DTR, P = f.PCA(DTR, PCA_m)  # fit PCA to training set

            # features expansion on training dataset, after PCA
            DTR_XXT = np.apply_along_axis(vec_xxT, 0, DTR)
            DTR = np.vstack([DTR_XXT, DTR])

            if PCA_m != None:
                # transform test samples according to P from PCA on dataset
                DTE = np.dot(P.T, DTE)

            # feature expansion on eval data, after PCA
            DTE_XXT = np.apply_along_axis(vec_xxT, 0, DTE)
            DTE = np.vstack([DTE_XXT, DTE])

            if (pi_prior == "Dataset" or pi_prior == "Fold"):
                # use dataset prior to train weighted logreg
                prior = d_prior
            else:
                # use remaining prior from list
                prior = pi_prior
            
            # compute scores on test set
            scores = models_llr.logistic_regression(DTR, LTR, l, DTE, LTE, prior, cal=True)

            # feedback print
            print(f"Computed PCA={PCA_m}, Lambda={l} ...")

            # compute minDCF for the current logreg(prior pi, labda) with/without PCA for the 2 working points
            minDCF = np.zeros(2)
            for i, pi in enumerate(pi_list):
                minDCF[i] = ev.compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
            # compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
            min_Cprim = np.round((minDCF[0] + minDCF[1]) / 2, 3)
            Cprim_list = np.hstack((Cprim_list, min_Cprim))
            # add current result to table
            results.add_row([pi_prior, PCA_m, l, np.round(minDCF[0], 3), np.round(minDCF[1], 3), min_Cprim])

        # feedback print
        print(f"Cprim values for PCA={PCA_m}: {Cprim_list}")
        # once computed Cprim for PCA_m, prior and different lambdas plot the graph
        ax.plot(l_list, Cprim_list, label =f'PCA-{PCA_m}')
        
    plt.legend()    
    #fig.savefig(f"Results/QuadLogReg_results/QuadLogReg_Prior{pi_prior}.png", dpi = 200)
    plt.show()

# print and save as txt the results table
print(results)
data = results.get_string()

with open('Results/QuadLogReg_results/QuadLogReg_ResultsTable.txt', 'w') as file:
    file.write(data)
