import scipy as sp
import numpy as np    
import threading
import evaluation
from Models.discriminative import * 
from Models.gaussians import *
from Models.gmm import *
from Models.svm import *
import sys

# Define the paths for the output files
mvg_output_file = '../Out/mvg_output.txt'
nb_output_file = '../Out/nb_output.txt'
tmvg_output_file = '../Out/tmvg_output.txt'
tnb_output_file = '../Out/tnb_output.txt'
logreg_output_file = '../Out/logreg_output.txt'
svm_output_file = '../Out/svm_output.txt'
gmm_output_file = '../Out/gmm_output.txt'

def PCA(D,m): # m = leading eigenvectors
    N = len(D)
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = N ** -1 * np.dot(DC, DC.T)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    return DP

def LDA(D,L,m,N_classes = 2):
    N = len(D[0])
    S_W = 0
    S_B = 0
    mu = np.array(np.mean(D, 1))

    for i in range(N_classes):
        D_class = D[:, L == i]
        n_c = len(D_class[0])
        mu_class = np.array(np.mean(D_class, 1))
        DC_class = D_class - vcol(mu_class)
        C_class = n_c ** -1 * np.dot(DC_class, DC_class.T)

        S_W += C_class * n_c
        S_B += n_c * np.dot(vcol(mu_class) - vcol(mu), (vcol(mu_class) - vcol(mu)).T)
    S_W = S_W / N
    S_B = S_B / N

    s, U = sp.linalg.eigh(S_B, S_W)
    W = U[:, ::-1][:, 0:m]

    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]

    DP = np.dot(W.T, D)

    return DP

K = 5
def MVG_kfold_wrapper(D, L):
    # print("MVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_MVG)))
    print("----------MVG-------------")
    original_stdout = sys.stdout
    with open(mvg_output_file, 'w') as f:
        sys.stdout = f
        
        for m in range(2,11):
            DP = PCA(D,m)
            print('PCA = {}'.format(m))
            print('minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_MVG)[2]))
        
        # Restore the original stdout
        sys.stdout = original_stdout

def NB_kfold_wrapper(D, L):
    # print("NB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_NaiveBayes)))
    print("----------NB-------------")
    original_stdout = sys.stdout
    with open(nb_output_file, 'w') as f:
        sys.stdout = f
            
        for m in range(2,11):
            DP = PCA(D,m)
            print('PCA = {}'.format(m))
            print('minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_NaiveBayes)[2]))
            
        # Restore the original stdout
        sys.stdout = original_stdout

def TMVG_kfold_wrapper(D, L):
    # print("TMVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedMVG)))
    print("----------TMVG-------------")
    
    original_stdout = sys.stdout
    with open(tmvg_output_file, 'w') as f:
        sys.stdout = f
    
        for m in range(2,11):
            DP = PCA(D,m)
            print('PCA = {}'.format(m))
            print('minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_TiedMVG)[2]))
        
        # Restore the original stdout
        sys.stdout = original_stdout

def TNB_kfold_wrapper(D, L):
    # print("TNB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedNaiveBayes)))
    print("----------TNB-------------")
    
    original_stdout = sys.stdout
    with open(tnb_output_file, 'w') as f:
        sys.stdout = f
        
        for m in range(2,11):
            DP = PCA(D,m)
            print('PCA = {}'.format(m))
            print('minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_TiedNaiveBayes)[2]))

    
        # Restore the original stdout
        sys.stdout = original_stdout

def logreg_kfold_wrapper(D, L):
    original_stdout = sys.stdout
    with open(logreg_output_file, 'w') as f:
        sys.stdout = f
    
        PCA_dim =  ['No PCA'] + [aa for aa in range(2, 10)]
        lam = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 100, 1000, 10000]
        for m in PCA_dim:
            if m == 'No PCA':
                print('-- Logistic Regression -- (no PCA)')
            else:
                print('-- Logistic Regression -- (PCA = {})'.format(m))
            for l in lam:
                    if m == 'No PCA':
                        _, _, minDCF, l = logreg_wrapper(D, L, l)
                        print('LogReg -- Lambda = {} -- minDCF: {}'.format(l, minDCF))
                        _, _, minDCF, l = QUAD_log_reg(D, L, l)
                        print('QUADLogReg -- Lambda = {} -- minDCF: {}'.format(l, minDCF))
                    else:
                        DP = PCA(D, m)
                        _, _, minDCF, l = logreg_wrapper(DP, L, l)
                        print('LogReg -- Lambda = {} -- minDCF: {}'.format(l, minDCF))
                        _, _, minDCF, l = QUAD_log_reg(DP, L, l)
                        print('QUADLogReg -- Lambda = {} -- minDCF: {}'.format(l, minDCF))
        # Restore the original stdout
        sys.stdout = original_stdout

def SVM_wrapper(D, L):
    original_stdout = sys.stdout
    with open(logreg_output_file, 'w') as f:
        sys.stdout = f
    
        cs = [10**-5, 2*(10**-5), 5*(10**-5)]
        D = np.append(D, np.ones((1,D.shape[1])),axis=0)
        #polynomial
        for c in cs:
            for mul in [1, 10, 100, 1000]:
                c_val = c * mul
                pol_kern = Kernel(d=2)
                svm = SVM(D, L, c_val, pol_kern.polynomial)
                print("c= "+str(c_val)+" poly("+str(2)+")")
                svm.exec()

                pol_kern = Kernel(d=3)
                svm = SVM(D, L, c_val, pol_kern.polynomial)
                print("c= "+str(c_val)+" poly("+str(3)+")")
                svm.exec()

        print("------------------")
        #rbf
        for c in cs:
            for mul in [1, 10, 100, 1000]:
                c_val = c * mul
                
                rbf_kern = Kernel(g=2)
                svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
                print("c= "+str(c_val)+" rbf("+str(2)+")")
                svm.exec()
                
                rbf_kern = Kernel(g=3)
                svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
                print("c= "+str(c_val)+" rbf("+str(3)+")")
                svm.exec()
                
                rbf_kern = Kernel(g=4)
                svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
                print("c= "+str(c_val)+" rbf("+str(4)+")")
                svm.exec()
                
                rbf_kern = Kernel(g=5)
                svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
                print("c= "+str(c_val)+" rbf("+str(5)+")")
                svm.exec()
        
        # Restore the original stdout
        sys.stdout = original_stdout
    """ #linear
    for c in cs:
        for mul in [1, 10, 100, 1000]:
            c = c * mul
            H = matrix_H_kernel(D, L, linear)
            SVM(D, L, H, c, linear) """
       
def GMM_wrapper(D, L):
     original_stdout = sys.stdout
    with open(logreg_output_file, 'w') as f:
        sys.stdout = f
   
        G = [1,2,4,8,16]
        functions = [GMM_EM, GMM_EM_diag, GMM_EM_tied]

        for g in G:
            print('------ g = {} ------'.format(g))
            for f in functions:
                err, minDCF = Kfold_cross_validation_GMM(D, L, K, g, f)
                # print("error: {}, using function: {} ".format(err,f.__name__))
                # print(pred[np.argmin(err)])

                print('function = {}, minDCF: {}'.format(f.__name__,minDCF))

        # Restore the original stdout
        sys.stdout = original_stdout



