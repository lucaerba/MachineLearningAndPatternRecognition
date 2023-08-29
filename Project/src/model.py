import scipy as sp
import numpy as np    
import threading
import evaluation
from prettytable import PrettyTable
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
                        # _, _, minDCF, l = logreg_wrapper(D, L, l)
                        # print('LogReg -- Lambda = {} -- minDCF: {}'.format(l, minDCF))
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
    with open(svm_output_file, 'w') as f:
        sys.stdout = f
        
        PCA_dim =  ['No PCA'] + [aa for aa in range(2, 10)]
        for m in PCA_dim:
            if m == "No PCA":
                print(" ---------------------- NO PCA")
                DP = D
            else:
                print(" ---------------------- PCA({})".format(m))
                DP = PCA(D, m)

            print("-------- Linear ---------")

            # linear
            cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            for c in cs:
                print(f' ----  c = {c}')
                for K in [1, 10, 100, 1000]:
                    svm = SVM(DP, L, c, K, Kernel.linear)
                    print(" --- K = {}".format(K))
                    for pi in [0.1, 0.5]:
                        for C_fp in [1, 10]:
                            svm.exec(pi=pi, C_fp=C_fp)

            print("-------- Polynomial ----------")

            #polynomial
            for c_val in cs:
                print(f' ----  c = {c}')
                for K in [1, 10, 100, 1000]:
                    print('---- d = 2 ----')
                    pol_kern = Kernel(d=2)
                    svm = SVM(DP, L, c_val, K, pol_kern.polynomial)
                    print(" --- K = {}".format(K))
                    for pi in [0.1, 0.5]:
                        for C_fp in [1, 10]:
                            svm.exec(pi=pi, C_fp=C_fp)

                for K in [1, 10, 100, 1000]:
                    print('---- d = 3 ----')
                    pol_kern = Kernel(d=3)
                    svm = SVM(DP, L, c_val, K, pol_kern.polynomial)
                    print(" --- K = {}".format(K))
                    for pi in [0.1, 0.5]:
                        for C_fp in [1, 10]:
                            svm.exec(pi=pi, C_fp=C_fp)

            print("-------- RBF ----------")
            #rbf
            for c_val in cs:
                print(f' ----  c = {c}')

                for K in [1, 10, 100, 1000]:
                    print(" --- K = {}".format(K))
                    for g in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
                        rbf_kern = Kernel(g)
                        svm = SVM(DP, L, c_val, K, rbf_kern.rbf_kernel)
                        print(f'-- g = {g}')
                        for pi in [0.1, 0.5]:
                            for C_fp in [1, 10]:
                                svm.exec(pi=pi, C_fp=C_fp)
        # Restore the original stdout
        sys.stdout = original_stdout

gmm_table = PrettyTable()
gmm_table.field_names = ['PCA', 'g (target)', 'g (NON target)', 'function', 'Working Point', 'minDCF', 'C_prim']
       
def GMM_wrapper(D, L):
    original_stdout = sys.stdout
    Cprim_selected = 1
    Working_Points = [(0.1, 1, 1), (0.5, 1, 1), (0.1, 1, 10)]
    with open(gmm_output_file, 'w') as f:
        sys.stdout = f

        PCA_dim =  ['No PCA'] + [aa for aa in range(2, 10)]
        for m in PCA_dim:
            if m == "No PCA":
                DP = D
            else:
                DP = PCA(D, m)
               
            G = [1,2,4,8,16]
            functions = [GMM_EM, GMM_EM_diag, GMM_EM_tied]

            for g_target in G:
                for g_NONtarget in G:
                    for f in functions:
                        _, minDCF0 = Kfold_cross_validation_GMM(DP, L, K, g_target, g_NONtarget, func=f)

                        gmm_table.add_row([m, g_target, g_NONtarget, f.__name__, (0.1,1,1), minDCF0, '-'])

                        _, minDCF1 = Kfold_cross_validation_GMM(DP, L, K, g_target, g_NONtarget,
                                                                 pi=0.5, func=f)
                        gmm_table.add_row(['-', '-', '-', '-', (0.5,1,1), minDCF1, '-'])

                        _, minDCF2 = Kfold_cross_validation_GMM(DP, L, K, g_target, g_NONtarget,
                                                                 C_fp=10, func=f)
                        gmm_table.add_row(['-', '-', '-', '-', (0.1,1,10), minDCF2, '-'])

                        C_prim = np.mean([minDCF0,minDCF1,minDCF2])
                        gmm_table.add_row(['-', '-', '-', '-', '-', '-', C_prim])

                        tot_solutions = [C_prim,Cprim_selected]
                        ind = np.argmin(tot_solutions)
                        if ind == 0:
                            m_best = m
                            g_target_best = g_target
                            g_NONtarget_best = g_NONtarget
                            f_best = f.__name__
                            C_prim_best = np.min(tot_solutions)

    print(f'BEST MODEL --> PCA : {m_best} -- g_target : {g_target_best} '
          f'-- g_NONtarget : {g_NONtarget_best} -- func : {f_best} '
          f'-- C_prim: {C_prim_best}')
    print(gmm_table)
    # Restore the original stdout
    sys.stdout = original_stdout



