"""This script performs training on train.txt and evaluation on test.txt for 
the best models chosen in validation phase with their best config, min and actual DCF are computed on test set

"""
import sys
sys.path.append("../")
from prettytable import PrettyTable
from validation import models_llr
import evaluation as ev
import Functions as f
import numpy as np


#import datetime

# set results table and final output
results = PrettyTable()
results.align = "r"
results.field_names = ["Model", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)",
                       "min Cprim", "actDCF (pi = 0.1)", "actDCF (pi = 0.5)", "act Cprim"]
out_txt = "Evaluation phase, following results are computed on Evaluation set (Test.txt) after training of models performed on training set (Train.txt)\n"

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
# add current result to table
results.add_row(["GMM 1F-64T Z-norm", np.round(minDCF[0], 3), np.round(minDCF[1], 3),
                min_Cprim, np.round(actDCF[0], 3), np.round(actDCF[1], 3), act_Cprim])

plt = ev.bayes_error_plot_comparison(scores, LTE, "GMM")

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
# add current result to table
results.add_row([f"RBF SVM Gamma:{gamma} PCA{PCA} calibration:{prior}, C:{C},K:{K}", np.round(minDCF[0], 3), np.round(minDCF[1], 3),
                min_Cprim, np.round(actDCF[0], 3), np.round(actDCF[1], 3), act_Cprim])
# add bayes plot 
plt = ev.bayes_error_plot_comparison(calibrated_scores, C_LTE, "SVM RBF", plot = plt, color='b')
plt.ylim([0, 1])
plt.xlim([-3, 3])
plt.grid()
plt.legend()
plt.savefig("Results/BayesPlot_Eval_Best_ValidationModels.png", dpi = 200)
plt.show()
print("SVM RBF done!")

print(results)
out_txt += results.get_string() + "\n Latex version:\n" + results.get_latex_string() 
with open('Results/Eval_Best_ValidationModels_Results.txt', 'w') as file:
    file.write(out_txt)

