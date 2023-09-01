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
import time
import os

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

MVG_table = PrettyTable()
MVG_table.field_names = ['PCA', 'Type', 'Working Point', 'minDCF', 'C_prim']

def MVG_kfold_wrapper(D, L):
    original_stdout = sys.stdout
    PCA_dim = ['No PCA'] + [aa for aa in range(2, D.shape[0])]
    Cprim_selected = 1
    with open(mvg_output_file, 'w') as f:
        sys.stdout = f

        for m in PCA_dim:
            t = time.time()
            sys.stdout = original_stdout
            print(f'starting PCA: {m}')
            sys.stdout = f
            if m == 'No PCA':
                DP = D
            else:
                DP = PCA(D, m)
            minDCF0 = Kfold_cross_validation(DP, L, K, func=score_matrix_MVG)[2]
            MVG_table.add_row([m, 'MVG', (0.1, 1, 1), minDCF0, '-'])
            minDCF1 = Kfold_cross_validation(DP, L, K, func=score_matrix_MVG, pi=0.5)[2]
            MVG_table.add_row([m, 'MVG', (0.5, 1, 1), minDCF1, '-'])

            C_prim = np.mean([minDCF0, minDCF1])
            MVG_table.add_row(['-', '-', '-', '-', C_prim])

            if C_prim < Cprim_selected:
                m_best = m
                C_prim_best = C_prim
                Cprim_selected = C_prim_best

            elapsed_time = time.time() - t
            sys.stdout = original_stdout
            print(f'finished PCA: {m}, elapsed time : {elapsed_time} s')
            sys.stdout = f

        print(f'MVG BEST --> PCA : {m_best} '
              f'-- C_prim: {C_prim_best}')
        print(MVG_table)
        # Restore the original stdout
        sys.stdout = original_stdout

NB_table = PrettyTable()
NB_table.field_names = ['PCA', 'Type', 'Working Point', 'minDCF', 'C_prim']
def NB_kfold_wrapper(D, L):
    original_stdout = sys.stdout
    PCA_dim = ['No PCA'] + [aa for aa in range(2, D.shape[0])]
    Cprim_selected = 1
    with open(nb_output_file, 'w') as f:
        sys.stdout = f

        for m in PCA_dim:
            t = time.time()
            sys.stdout = original_stdout
            print(f'starting PCA: {m}')
            sys.stdout = f
            if m == 'No PCA':
                DP = D
            else:
                DP = PCA(D, m)
            minDCF0 = Kfold_cross_validation(DP, L, K, func=score_matrix_NaiveBayes)[2]
            NB_table.add_row([m, 'NB', (0.1, 1, 1), minDCF0, '-'])
            minDCF1 = Kfold_cross_validation(DP, L, K, func=score_matrix_NaiveBayes, pi=0.5)[2]
            NB_table.add_row([m, 'NB', (0.5, 1, 1), minDCF1, '-'])

            C_prim = np.mean([minDCF0, minDCF1])
            NB_table.add_row(['-', '-', '-', '-', C_prim])

            if C_prim < Cprim_selected:
                m_best = m
                C_prim_best = C_prim
                Cprim_selected = C_prim_best

            elapsed_time = time.time() - t
            sys.stdout = original_stdout
            print(f'finished PCA: {m}, elapsed time : {elapsed_time} s')
            sys.stdout = f

        print(f'NB BEST --> PCA : {m_best} '
              f'-- C_prim: {C_prim_best}')
        print(NB_table)
        # Restore the original stdout
        sys.stdout = original_stdout

TMVG_table = PrettyTable()
TMVG_table.field_names = ['PCA', 'Type', 'Working Point', 'minDCF', 'C_prim']
def TMVG_kfold_wrapper(D, L):
    original_stdout = sys.stdout
    PCA_dim = ['No PCA'] + [aa for aa in range(2, D.shape[0])]
    Cprim_selected = 1
    with open(tmvg_output_file, 'w') as f:
        sys.stdout = f

        for m in PCA_dim:
            t = time.time()
            sys.stdout = original_stdout
            print(f'starting PCA: {m}')
            sys.stdout = f
            if m == 'No PCA':
                DP = D
            else:
                DP = PCA(D, m)
            minDCF0 = Kfold_cross_validation(DP, L, K, func=score_matrix_TiedMVG)[2]
            TMVG_table.add_row([m, 'TMVG', (0.1, 1, 1), minDCF0, '-'])
            minDCF1 = Kfold_cross_validation(DP, L, K, func=score_matrix_TiedMVG, pi=0.5)[2]
            TMVG_table.add_row([m, 'TMVG', (0.5, 1, 1), minDCF1, '-'])

            C_prim = np.mean([minDCF0, minDCF1])
            TMVG_table.add_row(['-', '-', '-', '-', C_prim])

            if C_prim < Cprim_selected:
                m_best = m
                C_prim_best = C_prim
                Cprim_selected = C_prim_best

            elapsed_time = time.time() - t
            sys.stdout = original_stdout
            print(f'finished PCA: {m}, elapsed time : {elapsed_time} s')
            sys.stdout = f

        print(f'TMVG BEST --> PCA : {m_best} '
              f'-- C_prim: {C_prim_best}')
        print(TMVG_table)
        # Restore the original stdout
        sys.stdout = original_stdout

TNB_table = PrettyTable()
TNB_table.field_names = ['PCA', 'Type', 'Working Point', 'minDCF', 'C_prim']

def TNB_kfold_wrapper(D, L):
    original_stdout = sys.stdout
    PCA_dim = ['No PCA'] + [aa for aa in range(2, D.shape[0])]
    Cprim_selected = 1
    with open(tnb_output_file, 'w') as f:
        sys.stdout = f

        for m in PCA_dim:
            t = time.time()
            sys.stdout = original_stdout
            print(f'starting PCA: {m}')
            sys.stdout = f
            if m == 'No PCA':
                DP = D
            else:
                DP = PCA(D, m)
            minDCF0 = Kfold_cross_validation(DP, L, K, func=score_matrix_TiedNaiveBayes)[2]
            TNB_table.add_row([m, 'TNB', (0.1, 1, 1), minDCF0, '-'])
            minDCF1 = Kfold_cross_validation(DP, L, K, func=score_matrix_TiedNaiveBayes, pi=0.5)[2]
            TNB_table.add_row([m, 'TNB', (0.5, 1, 1), minDCF1, '-'])

            C_prim = np.mean([minDCF0, minDCF1])
            TNB_table.add_row(['-', '-', '-', '-', C_prim])

            if C_prim < Cprim_selected:
                m_best = m
                C_prim_best = C_prim
                Cprim_selected = C_prim_best

            elapsed_time = time.time() - t
            sys.stdout = original_stdout
            print(f'finished PCA: {m}, elapsed time : {elapsed_time} s')
            sys.stdout = f

        print(f'TNB BEST --> PCA : {m_best} '
              f'-- C_prim: {C_prim_best}')
        print(TNB_table)
        # Restore the original stdout
        sys.stdout = original_stdout

##############################################################################################################################################################################
####################################### LogReg ##################################################################################################################################
##############################################################################################################################################################################

logreg_table = PrettyTable()
logreg_table.field_names = ['PCA', 'Type', 'lambda', 'Working Point', 'minDCF', 'C_prim']


def logreg_kfold_wrapper(D, L):
    original_stdout = sys.stdout
    Cprim_selected_lin = 1
    Cprim_selected_quad = 1
    with open(logreg_output_file, 'w') as f:
        sys.stdout = f
    
        PCA_dim =  ['No PCA'] + [aa for aa in range(2, D.shape[0])]
        lam = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 100, 1000, 10000]
        for m in PCA_dim:
            t = time.time()
            sys.stdout = original_stdout
            print(f'starting PCA: {m}')
            sys.stdout = f
            if m == 'No PCA':
                DP = D
            else:
                DP = PCA(D, m)
            for l in lam:
                    _, _, minDCF0, l = logreg_wrapper(DP, L, l)
                    logreg_table.add_row([m, 'Linear', l, (0.1, 1, 1), minDCF0, '-'])

                    _, _, minDCF1, l = logreg_wrapper(DP, L, l, pi=0.5)
                    logreg_table.add_row([m, 'Linear', l, (0.5, 1, 1), minDCF1, '-'])

                    _, _, minDCF2, l = logreg_wrapper(DP, L, l, C_fp=10)
                    logreg_table.add_row([m, 'Linear', l, (0.1, 1, 10), minDCF2, '-'])

                    C_prim = np.mean([minDCF0, minDCF1, minDCF2])
                    logreg_table.add_row(['-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_lin:
                        m_best_lin = m
                        l_best_lin = l
                        C_prim_best_lin = C_prim
                        Cprim_selected_lin = C_prim_best_lin

                    _, _, minDCF0, l = QUAD_log_reg(DP, L, l)
                    logreg_table.add_row([m, 'Quadratic', l, (0.1, 1, 1), minDCF0, '-'])

                    _, _, minDCF1, l = QUAD_log_reg(DP, L, l, pi=0.5)
                    logreg_table.add_row([m, 'Quadratic', l, (0.5, 1, 1), minDCF1, '-'])

                    _, _, minDCF2, l = QUAD_log_reg(DP, L, l, C_fp=10)
                    logreg_table.add_row([m, 'Quadratic', l, (0.1, 1, 10), minDCF2, '-'])

                    C_prim = np.mean([minDCF0, minDCF1, minDCF2])
                    logreg_table.add_row(['-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_quad:
                        m_best_quad = m
                        l_best_quad = l
                        C_prim_best_quad = C_prim
                        Cprim_selected_quad = C_prim_best_quad

            elapsed_time = time.time() - t
            sys.stdout = original_stdout
            print(f'finished PCA: {m}, elapsed time : {elapsed_time} s')
            sys.stdout = f

        print(f'Lin BEST --> PCA : {m_best_lin} -- l_best : {l_best_lin} '
              f'-- C_prim: {C_prim_best_lin}')
        print(f'Quad BEST --> PCA : {m_best_quad} -- l_best : {l_best_quad} '
              f'-- C_prim: {C_prim_best_quad}')
        print(logreg_table)

        # Restore the original stdout
        sys.stdout = original_stdout



##############################################################################################################################################################################
####################################### SVM ##################################################################################################################################
##############################################################################################################################################################################

svm_table = PrettyTable()
svm_table.field_names = ['PCA', 'Kernel', 'c', 'degree', 'gamma', 'K', 'Working Point', 'minDCF', 'C_prim']

def SVM_wrapper(D, L):
    original_stdout = sys.stdout
    Cprim_selected_lin = 1
    Cprim_selected_poly2 = 1
    Cprim_selected_poly3 = 1
    Cprim_selected_rbf = 1
    with open(svm_output_file, 'w') as f:
        sys.stdout = f
        
        PCA_dim =  ['No PCA'] + [aa for aa in range(2, D.shape[0])]
        for m in PCA_dim:
            t = time.time()
            sys.stdout = original_stdout
            print(f'starting PCA: {m}')
            sys.stdout = f
            if m == "No PCA":
                DP = D
            else:
                DP = PCA(D, m)

            # linear
            cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            for c in cs:
                for K in [1, 10, 100, 1000]:
                    svm = SVM(DP, L, c, K, Kernel.linear)

                    minDCF0 = svm.exec()
                    svm_table.add_row([m, 'Linear', c, '-', '-', K, (0.1, 1, 1), minDCF0, '-'])

                    minDCF1 = svm.exec(pi=0.5)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                    minDCF2 = svm.exec(C_fp=10)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.1,1,10), minDCF0, '-'])

                    C_prim = np.mean([minDCF0, minDCF1, minDCF2])
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_lin:
                        m_best_lin = m
                        c_best_lin = c
                        K_best_lin = K
                        C_prim_best_lin = C_prim
                        Cprim_selected_lin = C_prim_best_lin

            #polynomial

            for c_val in cs:
                for K in [1, 10, 100, 1000]:
                    pol_kern = Kernel(d=2)
                    svm = SVM(DP, L, c_val, K, pol_kern.polynomial)

                    minDCF0 = svm.exec()
                    svm_table.add_row([m, 'Poly', c_val, 2, '-', K, (0.1, 1, 1), minDCF0, '-'])

                    minDCF1 = svm.exec(pi=0.5)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                    minDCF2 = svm.exec(C_fp=10)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.1,1,10), minDCF0, '-'])

                    C_prim = np.mean([minDCF0, minDCF1, minDCF2])
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_poly2:
                        m_best_poly2 = m
                        c_best_poly2 = c_val
                        d_best_poly2 = 2
                        K_best_poly2 = K
                        C_prim_best_poly2 = C_prim
                        Cprim_selected_poly2 = C_prim_best_poly2

                for K in [1, 10, 100, 1000]:
                    pol_kern = Kernel(d=3)
                    svm = SVM(DP, L, c_val, K, pol_kern.polynomial)

                    minDCF0 = svm.exec()
                    svm_table.add_row([m, 'Poly', c_val, 3, '-', K, (0.1, 1, 1), minDCF0, '-'])

                    minDCF1 = svm.exec(pi=0.5)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                    minDCF2 = svm.exec(C_fp=10)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.1,1,10), minDCF0, '-'])

                    C_prim = np.mean([minDCF0, minDCF1, minDCF2])
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_poly3:
                        m_best_poly3 = m
                        c_best_poly3 = c_val
                        d_best_poly3 = 3
                        K_best_poly3 = K
                        C_prim_best_poly3 = C_prim
                        Cprim_selected_poly3 = C_prim_best_poly3

            #rbf
            for c_val in cs:
                for K in [1, 10, 100, 1000]:
                    for g in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
                        rbf_kern = Kernel(g)
                        svm = SVM(DP, L, c_val, K, rbf_kern.rbf_kernel)

                        minDCF0 = svm.exec()
                        svm_table.add_row([m, 'RBF', c_val, '-', g, K, (0.1, 1, 1), minDCF0, '-'])

                        minDCF1 = svm.exec(pi=0.5)
                        svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                        minDCF2 = svm.exec(C_fp=10)
                        svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.1, 1, 10), minDCF0, '-'])

                        C_prim = np.mean([minDCF0, minDCF1, minDCF2])
                        svm_table.add_row(['-', '-', '-', '-', '-', '-', '-', '-', C_prim])

                        if C_prim < Cprim_selected_rbf:
                            m_best_RBF = m
                            c_best_RBF = c_val
                            g_best_RBF = g
                            K_best_RBF = K
                            C_prim_best_RBF = C_prim
                            Cprim_selected_rbf = C_prim_best_RBF

            elapsed_time = time.time() - t
            sys.stdout = original_stdout
            print(f'finished PCA: {m}, elapsed time : {elapsed_time} s')
            sys.stdout = f

        print(f'RBF BEST --> PCA : {m_best_RBF} -- c_best : {c_best_RBF} '
              f'-- gamma_best : {g_best_RBF} -- K_best : {K_best_RBF} '
              f'-- C_prim: {C_prim_best_RBF}')
        print(f'Poly (dim3) BEST --> PCA : {m_best_poly3} -- c_best : {c_best_poly3} '
              f'-- d_best : {d_best_poly3} -- K_best : {K_best_poly3} '
              f'-- C_prim: {C_prim_best_poly3}')
        print(f'Poly (dim2) BEST --> PCA : {m_best_poly2} -- c_best : {c_best_poly2} '
              f'-- d_best : {d_best_poly2} -- K_best : {K_best_poly2} '
              f'-- C_prim: {C_prim_best_poly2}')
        print(f'Lin BEST --> PCA : {m_best_lin} -- c_best : {c_best_lin} '
              f'-- K_best : {K_best_lin} '
              f'-- C_prim: {C_prim_best_lin}')
        print(svm_table)

        # Restore the original stdout
        sys.stdout = original_stdout


##############################################################################################################################################################################
####################################### GMM ##################################################################################################################################
##############################################################################################################################################################################


gmm_table = PrettyTable()
gmm_table.field_names = ['PCA', 'g (target)', 'g (NON target)', 'function', 'Working Point', 'minDCF', 'C_prim']
       
def GMM_wrapper(D, L):
    original_stdout = sys.stdout
    Cprim_selected = 1
    # Working_Points = [(0.1, 1, 1), (0.5, 1, 1), (0.1, 1, 10)]
    with open(gmm_output_file, 'w') as f:
        sys.stdout = f

        PCA_dim =  ['No PCA'] + [aa for aa in range(2, D.shape[0])]
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

                        C_prim = np.mean([minDCF0,minDCF1])
                        gmm_table.add_row(['-', '-', '-', '-', '-', '-', C_prim])

                        if C_prim < Cprim_selected:
                            m_best = m
                            g_target_best = g_target
                            g_NONtarget_best = g_NONtarget
                            f_best = f.__name__
                            C_prim_best = C_prim
                            Cprim_selected = C_prim_best

        print(f'BEST MODEL --> PCA : {m_best} -- g_target : {g_target_best} '
              f'-- g_NONtarget : {g_NONtarget_best} -- func : {f_best} '
              f'-- C_prim: {C_prim_best}')
        print(gmm_table)
        # Restore the original stdout
        sys.stdout = original_stdout



