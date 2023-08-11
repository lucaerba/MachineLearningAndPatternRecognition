"""This script performs training on train.txt and evaluation on test.txt 
The tested models are: 
    -SVM RBF chosen in validation config and best evaluation config
    - GMM chosen in validation config and best eval config (they are the same because chosen GMM config. is optimal also on test set)
The output will be a table which confronts the results in terms of min Cprim and act Cprim computed 
by these models both on evaluation set

"""
import sys
sys.path.append("../")
from validation import models_llr
import evaluation as ev
import Functions as f
import numpy as np



# load training and test data
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
print(
    f"Found {DTR.shape[1]} sample for training and {DTE.shape[1]} samples for test.")

# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

#####  GMM  #####

print("GMM started...")

# set GMM-LBG doublings and GMM version to use for both target and non-target class, number of components for GMM model will be 2^(doub_target/nonTarget)
version_target = "full"
doub_target = 0
version_nonTarget = "tied"
doub_nonTarget = 6

# Z-Norm
DTR, DTE = f.apply_Z_Norm(DTR, DTE)

# get training samples of class 0, train a GMM on them, compute SM for class 0 on test set
DTR0 = DTR[:, LTR == 0]
gmm_class0 = models_llr.GMM_LBG(DTR0, doub_nonTarget, version_nonTarget)
_, SM0 = models_llr.logpdf_GMM(DTE, gmm_class0)

# same for class 1
DTR1 = DTR[:, LTR == 1]
gmm_class1 = models_llr.GMM_LBG(DTR1, doub_target, version_target)
_, SM1 = models_llr.logpdf_GMM(DTE, gmm_class1)

# compute scores on test set
scores = SM1 - SM0

# compute minDCF and actualDCF for the working points
minDCF = np.zeros(2)
actDCF = np.zeros(2)
for i, pi in enumerate(pi_list):
    actDCF[i] = ev.compute_act_DCF(scores, LTE, pi, Cfn, Cfp)
    minDCF[i] = ev.compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
# compute Cprim (the 2 working points average minDCF)
min_Cprim = np.round((minDCF[0] + minDCF[1]) / 2, 3)
act_Cprim = np.round((actDCF[0] + actDCF[1]) / 2, 3)

out_str = f"GMM best validation config and best evaluation config. are the same.\nResults on evaluation set:\nmin Cprim: {min_Cprim}\tact Cprim: {act_Cprim}"


print("GMM done!")

##### SVM RBF #####

print("Starting SVM RBF")

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

# best RBF SVM configuration in validation phase with score calibration
PCA = 5  # number of dimension to keep in PCA
C = 10
K = 0.01
gamma = 0.1

DTR, P = f.PCA(DTR, PCA)  # fit PCA to training set
DTE = np.dot(P.T, DTE)    # apply PCA to test set

#get scores on test set
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
prior = 0.2
calibrated_scores = models_llr.logistic_regression(C_DTR, C_LTR, 0, C_DTE, C_LTE, prior, cal = True)

# compute minDCF and actDCF
minDCF = np.zeros(2)
actDCF = np.zeros(2)
for i, pi in enumerate(pi_list):
    actDCF[i] = ev.compute_act_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
    minDCF[i] = ev.compute_min_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
min_Cprim = np.round((minDCF[0] + minDCF[1]) / 2, 3)
act_Cprim = np.round((actDCF[0] + actDCF[1]) / 2, 3)
# print results
out_str += f"\nSVM RBF config chosen in validation phase results on evaluation set:\nmin Cprim: {min_Cprim}\tact Cprim: {act_Cprim}"


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

# best RBF SVM configuration from evaluation phase with score calibration
PCA = 5  # number of dimension to keep in PCA
C = 10
K = 0.1
gamma = 0.001

DTR, P = f.PCA(DTR, PCA)  # fit PCA to training set
DTE = np.dot(P.T, DTE)    # apply PCA to test set

#get scores on test set
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
prior = 0.2
calibrated_scores = models_llr.logistic_regression(C_DTR, C_LTR, 0, C_DTE, C_LTE, prior, cal = True)

# compute minDCF and actDCF
minDCF = np.zeros(2)
actDCF = np.zeros(2)
for i, pi in enumerate(pi_list):
    actDCF[i] = ev.compute_act_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
    minDCF[i] = ev.compute_min_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
min_Cprim = np.round((minDCF[0] + minDCF[1]) / 2, 3)
act_Cprim = np.round((actDCF[0] + actDCF[1]) / 2, 3)
# print results
out_str += f"\nSVM RBF optimal config from evaluation phase results on evaluation set:\nmin Cprim: {min_Cprim}\tact Cprim: {act_Cprim}"
print("SVM RBF done!")

print(out_str)


