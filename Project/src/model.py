import scipy as sp
import numpy as np    
import threading
import evaluation
import matplotlib.pyplot as plt
from itertools import combinations
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
# svm_calibration_output = '../Out/svm_calibration_output.txt'
gmm_output_file = '../Out/gmm_output.txt'
logreg_output_file_Znorm = '../FINALoutputs/logreg_output_Znorm.txt'
best_output_file = '../FINALoutputs/best_models.txt'
fusion_output_file = '../FINALoutputs/fusion_models.txt'


def PCA(D,m, m_eig=False): # m = leading eigenvectors
    N = len(D)
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = N ** -1 * np.dot(DC, DC.T)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    if m_eig == False:
        return DP
    else:
        return DP, P

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

                    C_prim = np.mean([minDCF0, minDCF1])
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

                    C_prim = np.mean([minDCF0, minDCF1])
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

##################################################### Logreg Z-norm ##########################################################################################################

logreg_Znorm_table = PrettyTable()
logreg_Znorm_table.field_names = ['PCA', 'Type', 'lambda', 'Working Point', 'minDCF', 'C_prim']
def logreg_Znorm_wrapper(D, L):
    original_stdout = sys.stdout
    Cprim_selected_quad = 1
    with open(logreg_output_file_Znorm, 'w') as f:
        sys.stdout = f

        PCA_dim = ['No PCA'] + [9,8,7]
        lam = [1e-6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10]
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

                _, _, minDCF0, l = QUAD_log_reg(DP, L, l, Znorm=True)
                logreg_Znorm_table.add_row([m, 'Quadratic', l, (0.1, 1, 1), minDCF0, '-'])

                _, _, minDCF1, l = QUAD_log_reg(DP, L, l, pi=0.5, Znorm=True)
                logreg_Znorm_table.add_row([m, 'Quadratic', l, (0.5, 1, 1), minDCF1, '-'])

                C_prim = np.mean([minDCF0, minDCF1])
                logreg_Znorm_table.add_row(['-', '-', '-', '-', '-', C_prim])

                if C_prim < Cprim_selected_quad:
                    m_best_quad = m
                    l_best_quad = l
                    C_prim_best_quad = C_prim
                    Cprim_selected_quad = C_prim_best_quad

            elapsed_time = time.time() - t
            sys.stdout = original_stdout
            print(f'finished PCA: {m}, elapsed time : {elapsed_time} s')
            sys.stdout = f

        print(f'Quad BEST --> PCA : {m_best_quad} -- l_best : {l_best_quad} '
              f'-- C_prim: {C_prim_best_quad}')
        print(logreg_Znorm_table)

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
        
        PCA_dim =  ['No PCA'] + [9,8,7,6]
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
            cs = [1e-4, 1e-3, 1e-2, 1e-1, 1]
            Ks = [0.1,1,10]
            for c in cs:
                for K in Ks:
                    svm = SVM(DP, L, c, K, Kernel(K=K).linear)

                    minDCF0 = svm.exec()
                    svm_table.add_row([m, 'Linear', c, '-', '-', K, (0.1, 1, 1), minDCF0, '-'])

                    minDCF1 = svm.exec(pi=0.5)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                    C_prim = np.mean([minDCF0, minDCF1])
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_lin:
                        m_best_lin = m
                        c_best_lin = c
                        K_best_lin = K
                        C_prim_best_lin = C_prim
                        Cprim_selected_lin = C_prim_best_lin

            #polynomial

            for c_val in cs:
                for K in Ks:
                    pol_kern = Kernel(d=2, K=K)
                    svm = SVM(DP, L, c_val, K, pol_kern.polynomial)

                    minDCF0 = svm.exec()
                    svm_table.add_row([m, 'Poly', c_val, 2, '-', K, (0.1, 1, 1), minDCF0, '-'])

                    minDCF1 = svm.exec(pi=0.5)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                    C_prim = np.mean([minDCF0, minDCF1])
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_poly2:
                        m_best_poly2 = m
                        c_best_poly2 = c_val
                        d_best_poly2 = 2
                        K_best_poly2 = K
                        C_prim_best_poly2 = C_prim
                        Cprim_selected_poly2 = C_prim_best_poly2

                for K in Ks:
                    pol_kern = Kernel(d=3, K=K)
                    svm = SVM(DP, L, c_val, K, pol_kern.polynomial)

                    minDCF0 = svm.exec()
                    svm_table.add_row([m, 'Poly', c_val, 3, '-', K, (0.1, 1, 1), minDCF0, '-'])

                    minDCF1 = svm.exec(pi=0.5)
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                    C_prim = np.mean([minDCF0, minDCF1])
                    svm_table.add_row(['-', '-', '-', '-', '-', '-', '-', '-', C_prim])

                    if C_prim < Cprim_selected_poly3:
                        m_best_poly3 = m
                        c_best_poly3 = c_val
                        d_best_poly3 = 3
                        K_best_poly3 = K
                        C_prim_best_poly3 = C_prim
                        Cprim_selected_poly3 = C_prim_best_poly3

            #rbf
            gs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            for c_val in cs:
                for K in Ks:
                    for g in gs:
                        rbf_kern = Kernel(g=g, K=K)
                        svm = SVM(DP, L, c_val, K, rbf_kern.rbf_kernel)

                        minDCF0 = svm.exec()
                        svm_table.add_row([m, 'RBF', c_val, '-', g, K, (0.1, 1, 1), minDCF0, '-'])

                        minDCF1 = svm.exec(pi=0.5)
                        svm_table.add_row(['-', '-', '-', '-', '-', '-', (0.5, 1, 1), minDCF1, '-'])

                        C_prim = np.mean([minDCF0, minDCF1])
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

############################################# BEST models ##############################################################################################################################

# actual DCF
best_table = PrettyTable()
best_table.field_names = ['model', 'Working Point', 'actDCF', 'minDCF', 'C_prim']

PCA_logreg = 7
PCA_MVG = 8
PCA_SVM = 9

lam = 1e-5

c = 1e-4
K = 10

g_target = 1
g_NONtarget = 8

def actual_DCF_and_scores(D, L):
    original_stdout = sys.stdout

    with open(best_output_file, 'w') as f:
        sys.stdout = f

        # MVG
        DP = PCA(D,PCA_MVG)
        actDCF0, scores_MVG0, minDCF0 = Kfold_cross_validation(DP, L, 5, func=score_matrix_MVG)
        best_table.add_row(['MVG', (0.1,1,1), actDCF0, minDCF0, '-'])
        actDCF1, scores_MVG1, minDCF1 = Kfold_cross_validation(DP, L, 5, func=score_matrix_MVG, pi=0.5)
        best_table.add_row(['MVG', (0.5, 1, 1), actDCF1, minDCF1, '-'])
        C_prim = np.mean([minDCF0, minDCF1])
        best_table.add_row(['-', '-', '-', '-', C_prim])

        # logreg
        DP = PCA(D, PCA_logreg)
        actDCF0, minDCF0, scores_logreg0, _ = QUAD_log_reg(DP, L, lam, Znorm=True)
        best_table.add_row(['logreg', (0.1, 1, 1), actDCF0, minDCF0, '-'])
        actDCF1, minDCF1, scores_logreg1, _ = QUAD_log_reg(DP, L, lam, pi=0.5, Znorm=True)
        best_table.add_row(['logreg', (0.5, 1, 1), actDCF1, minDCF1, '-'])
        C_prim = np.mean([minDCF0, minDCF1])
        best_table.add_row(['-', '-', '-', '-', C_prim])

        # SVM
        DP = PCA(D, PCA_SVM)
        pol_kern = Kernel(d=2, K=K)
        svm = SVM(DP, L, c, K, pol_kern.polynomial)
        actDCF0, minDCF0, scores_SVM0 = svm.exec()
        best_table.add_row(['SVM', (0.1, 1, 1), actDCF0, minDCF0, '-'])
        actDCF1, minDCF1, scores_SVM1 = svm.exec(pi=0.5)
        best_table.add_row(['SVM', (0.5, 1, 1), actDCF1, minDCF1, '-'])
        C_prim = np.mean([minDCF0, minDCF1])
        best_table.add_row(['-', '-', '-', '-', C_prim])

        # GMM
        DP = D
        actDCF0, minDCF0, scores_GMM0 = Kfold_cross_validation_GMM(DP, L, 5, g_target, g_NONtarget, func=GMM_EM)
        best_table.add_row(['GMM', (0.1, 1, 1), actDCF0, minDCF0, '-'])
        actDCF1, minDCF1, scores_GMM1 = Kfold_cross_validation_GMM(DP, L, 5, g_target, g_NONtarget, pi=0.5, func=GMM_EM)
        best_table.add_row(['GMM', (0.5, 1, 1), actDCF1, minDCF1, '-'])
        C_prim = np.mean([minDCF0, minDCF1])
        best_table.add_row(['-', '-', '-', '-', C_prim])

        print(best_table)
        print(f'MVG scores app1: {scores_MVG0}')
        print(f'MVG scores app2: {scores_MVG1}')
        print(f'logreg scores app1: {scores_logreg0}')
        print(f'logreg scores app2: {scores_logreg1}')
        print(f'SVM scores app1: {scores_SVM0}')
        print(f'SVM scores app2: {scores_SVM1}')
        print(f'GMM scores app1: {scores_GMM0}')
        print(f'GMM scores app2: {scores_GMM1}')

        sys.stdout = original_stdout



################################################## CALIBRATION ############################################################################

def calibration(D, L, K_fold=5, seed=1, model='MVG'):
    nSamp = int(D.shape[1] / K_fold)
    residuals = D.shape[1] - nSamp * K_fold
    sub_arr = np.ones((K_fold, 1)) * nSamp

    if residuals != 0:
        sub_arr = np.array([int(x + 1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    scores = np.array([])
    labels = np.array([])

    for i in range(K_fold):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i + 1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]


        if model == "MVG":
            DTR, P = PCA(DTR, PCA_MVG, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_gaussian(DTR, LTR, DTE)))
        if model == "logreg":
            DTR, P = PCA(DTR, PCA_logreg, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_logreg(DTR, LTR, DTE, lam)))
        if model == "SVM":
            DTR, P = PCA(DTR, PCA_SVM, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            pol_kern = Kernel(d=2, K=K)
            svm = SVM(DTR, L, c, K, pol_kern.polynomial)
            scores = np.hstack((scores, svm.scores(DTR, LTR, DTE)))

    idx_new = np.random.permutation(scores.shape[0])
    scores = scores[idx_new]
    labels = labels[idx_new]

    frac = 7/10
    DTR = vrow(scores[:int(scores.shape[0]* frac)])
    LTR = labels[:int(labels.shape[0] * frac)]
    DTE = vrow(scores[int(scores.shape[0] * frac):])
    LTE = labels[int(labels.shape[0] * frac):]

    return DTR, LTR, DTE, LTE


pi_calibration = 1e-1 * np.arange(1,10,1)
def calibration_wrapper(D,L):
    for p in pi_calibration:
        plt.figure()
        for i, model in enumerate(['MVG', 'logreg', 'SVM']):
            DTR, LTR, DTE, LTE = calibration(D, L, model=model)
            scores_cal, _, _ = logreg_wrapper(D, L, 0, pi=p, C_fn=1, C_fp=1, cal=True, DTR=DTR, LTR=LTR, DTE=DTE)
            evaluation.Bayes_error_plots(scores_cal, LTE, pi=p, model=model, i_col=i)


################################################## FUSION ######################################################################################################################################################

def best_scores(D, L, K_fold=5, seed=1, model='MVG'):
    nSamp = int(D.shape[1] / K_fold)
    residuals = D.shape[1] - nSamp * K_fold
    sub_arr = np.ones((K_fold, 1)) * nSamp

    if residuals != 0:
        sub_arr = np.array([int(x + 1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    scores = np.array([])
    labels = np.array([])

    for i in range(K_fold):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i + 1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]


        if model == "MVG":
            DTR, P = PCA(DTR, PCA_MVG, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_gaussian(DTR, LTR, DTE)))
        elif model == "logreg":
            DTR, P = PCA(DTR, PCA_logreg, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_logreg(DTR, LTR, DTE, lam)))
        elif model == "SVM":
            DTR, P = PCA(DTR, PCA_SVM, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            pol_kern = Kernel(d=2, K=K)
            svm = SVM(DTR, L, c, K, pol_kern.polynomial)
            scores = np.hstack((scores, svm.scores(DTR, LTR, DTE)))
        elif model == 'GMM':

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_gmm(DTR, LTR, DTE, g_target, g_NONtarget)))

    np.random.seed(seed+1)
    idx_new = np.random.permutation(scores.shape[0])
    scores = scores[idx_new]
    labels = labels[idx_new]

    frac = 7 / 10
    DTR = vrow(scores[:int(scores.shape[0] * frac)])
    LTR = labels[:int(labels.shape[0] * frac)]
    DTE = vrow(scores[int(scores.shape[0] * frac):])
    LTE = labels[int(labels.shape[0] * frac):]

    return DTR, LTR, DTE, LTE


fusion_table = PrettyTable()
fusion_table.field_names = ['models', 'actDCF (pi = 0.1)', 'actDCF (pi = 0.5)',
                            'minDCF (pi = 0.1)', 'minDCF (pi = 0.5)', 'C_prim']

def fusion_wrapper(D, L):
    original_stdout = sys.stdout

    with open(fusion_output_file, 'w') as f:
        sys.stdout = f
        models = np.array(['GMM', 'SVM', 'logreg', 'MVG'])
        DTR_gmm, LTR, DTE_gmm, LTE = best_scores(D,L,model='GMM')
        DTR_svm, _, DTE_svm, _ = best_scores(D, L, model='SVM')
        DTR_logreg, _, DTE_logreg, _ = best_scores(D, L, model='logreg')
        DTR_mvg, _, DTE_mvg, _ = best_scores(D, L, model='MVG')

        p = 0.5
        scores_fus_GMM, _, _ = logreg_wrapper(D, L, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR_gmm), LTR=LTR,
                                              DTE=vrow(DTE_gmm))
        scores_fus_SVM, _, _ = logreg_wrapper(D, L, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR_svm), LTR=LTR,
                                              DTE=vrow(DTE_svm))
        scores_fus_logreg, _, _ = logreg_wrapper(D, L, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                                 DTR=vrow(DTR_logreg), LTR=LTR,
                                                 DTE=vrow(DTE_logreg))
        scores_fus_MVG, _, _ = logreg_wrapper(D, L, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR_mvg), LTR=LTR,
                                              DTE=vrow(DTE_mvg))

        couples = list(combinations(models, 2))
        triplets = list(combinations(models, 3))

        all_combins = couples + triplets + [tuple(models)]

        for combin_i in all_combins:
            idx = [False, False, False, False]
            scores_fus = np.zeros(DTE_gmm.shape[1])
            if 'GMM' in combin_i:
                idx[0] = True
                scores_fus += np.array(scores_fus_GMM)
            if 'SVM' in combin_i:
                idx[1] = True
                scores_fus += np.array(scores_fus_SVM)
            if 'logreg' in combin_i:
                idx[2] = True
                scores_fus += np.array(scores_fus_logreg)
            if 'MVG' in combin_i:
                idx[3] = True
                scores_fus += np.array(scores_fus_MVG)

            minDCF0 = evaluation.minDCF(scores_fus, LTE, 0.1,1,1)
            actDCF0 = evaluation.Bayes_risk_normalized(scores_fus, LTE, 0.1, 1, 1)
            minDCF1 = evaluation.minDCF(scores_fus, LTE, 0.5, 1, 1)
            actDCF1 = evaluation.Bayes_risk_normalized(scores_fus, LTE, 0.5, 1, 1)
            C_prim = np.mean([minDCF0, minDCF1])
            fusion_table.add_row([models[idx], actDCF0, actDCF1, minDCF0, minDCF1, C_prim])

        print(fusion_table)
        sys.stdout = original_stdout


def scores_to_plot(D, L, K_fold = 5, seed = 1, model = 'MVG'):
    nSamp = int(D.shape[1] / K_fold)
    residuals = D.shape[1] - nSamp * K_fold
    sub_arr = np.ones((K_fold, 1)) * nSamp

    if residuals != 0:
        sub_arr = np.array([int(x + 1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    scores = np.array([])
    labels = np.array([])

    for i in range(K_fold):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i + 1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        if model == "MVG":
            DTR, P = PCA(DTR, PCA_MVG, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_gaussian(DTR, LTR, DTE)))

        elif model == "logreg":
            DTR, P = PCA(DTR, PCA_logreg, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_logreg(DTR, LTR, DTE, lam)))
        elif model == "SVM":
            DTR, P = PCA(DTR, PCA_SVM, m_eig=True)
            DTE = np.dot(P.T, DTE)

            labels = np.hstack((labels, LTE))
            pol_kern = Kernel(d=2, K=K)
            svm = SVM(DTR, L, c, K, pol_kern.polynomial)
            scores = np.hstack((scores, svm.scores(DTR, LTR, DTE)))
        elif model == 'GMM':

            labels = np.hstack((labels, LTE))
            scores = np.hstack((scores, scores_gmm(DTR, LTR, DTE, g_target, g_NONtarget)))

    return scores, labels

def DET_plot(D,L):
    models = np.array(['GMM', 'SVM', 'logreg', 'MVG'])
    colors = ['red', 'green', 'blue', 'fuchsia']

    plt.figure()
    for i, model in enumerate(models):
        scores, labels = scores_to_plot(D, L, model=model)
        evaluation.DET_curve(scores, labels, model=model, color=colors[i])
        if i == 3:
            plt.savefig('../Plots/DET_plot_BESTmodels.png', bbox_inches='tight')
            # plt.show(bbox_inches='tight')
            plt.close()