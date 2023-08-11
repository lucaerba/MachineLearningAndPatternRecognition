"""This script compare the best models of Validation phase with best configuration in a bayes plots
Bayes plot computed on scores from validation set!
"""
import sys
sys.path.append("../")
import numpy as np
import Functions as f
import evaluation as ev
import models_llr
from prettytable import PrettyTable
#import datetime

# load training data
D, L = f.loadData('../Train.txt')

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

# K-fold settings
K = 5 
N = int(D.shape[1]/K)
np.random.seed(0)
indexes = np.random.permutation(D.shape[1]) 

# ### Set and validate GMM ###

#PCA dimensions to use
PCA = [None]

# set GMM-LBG doublings and GMM version to use for both target and non-target class, number of components for GMM model will be 2^(doub_target/nonTarget)
version_target = "full"
doub_target_list = [0]
version_nonTarget = "tied"
doub_nonTarget_list = [6]

final_output = f"Target Class GMM: {version_target}, Non-Target Class GMM: {version_nonTarget}\n"

for doub_target in doub_target_list:
    for doub_nonTarget in doub_nonTarget_list:
        
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
        
#compute after k-fold the bayes plot 
plt = ev.bayes_error_plot_comparison(scores_pool, labels_pool, "GMM 1-F 64-T (Z-Norm)")
        
### Set and validate SVM RBF
# number of folds for K-fold
folds = 5 # can't use K for K-fold because in SVM K is already used
N = int(D.shape[1]/folds)
PCA = 5 #number of dimension to keep in PCA

np.random.seed(0)
indexes = np.random.permutation(D.shape[1]) 
#best RBF SVM configuration in validation phase with score calibration
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
prior = 0.2 # target prior
calibrated_scores = models_llr.logistic_regression(C_DTR, C_LTR, 0, C_DTE, C_LTE, prior, cal = True)
#compute minDCF and actDCF on calibrated vlaidation set
plt = ev.bayes_error_plot_comparison(calibrated_scores, C_LTE, "SVM RBF", plot = plt, color = 'b')
plt.ylim([0, 1])
plt.xlim([-3, 3])
plt.legend()
plt.savefig("BayesPlots/BayesPlot_Comparison.png", dpi = 200)
plt.show()

          
      