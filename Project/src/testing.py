import numpy as np
import evaluation
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from Models.discriminative import *
from Models.gaussians import *
from Models.gmm import *
from Models.svm import *
import sys
import time
import os

def PCA(D,m): # m = leading eigenvectors
    N = len(D)
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = N ** -1 * np.dot(DC, DC.T)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    return DP, P


PCA_logreg = 7
PCA_MVG = 8
PCA_SVM = 9

lam = 1e-5

c = 1e-4
K = 10

g_target = 1
g_NONtarget = 8
def scores_test(DTR_fin, DTE_fin, LTR_fin, LTE_fin, model='MVG', seed=0, calibration=False):

    if model == "MVG":
        DTR, P = PCA(DTR_fin, PCA_MVG)
        DTE = np.dot(P.T, DTE_fin)

        scores = scores_gaussian(DTR, LTR_fin, DTE)

        if calibration == False:
            return scores, LTE_fin
        else:
            np.random.seed(seed)
            idx_new = np.random.permutation(scores.shape[0])
            scores = scores[idx_new]
            labels = LTE_fin[idx_new]

            frac = 7 / 10
            DTR = vrow(scores[:int(scores.shape[0] * frac)])
            LTR = labels[:int(labels.shape[0] * frac)]
            DTE = vrow(scores[int(scores.shape[0] * frac):])
            LTE = labels[int(labels.shape[0] * frac):]
            scores, _, _ = logreg_wrapper(DTR_fin, LTR_fin, 0, pi=0.4, C_fn=1, C_fp=1, cal=True,
                                          DTR=vrow(DTR), LTR=LTR, DTE=DTE)
            return scores, LTE

    elif model == "logreg":
        DTR, P = PCA(DTR_fin, PCA_logreg)
        DTE = np.dot(P.T, DTE_fin)

        scores = scores_logreg(DTR, LTR_fin, DTE, lam)

        if calibration == False:
            return scores, LTE_fin
        else:
            np.random.seed(seed)
            idx_new = np.random.permutation(scores.shape[0])
            scores = scores[idx_new]
            labels = LTE_fin[idx_new]

            frac = 7 / 10
            DTR = vrow(scores[:int(scores.shape[0] * frac)])
            LTR = labels[:int(labels.shape[0] * frac)]
            DTE = vrow(scores[int(scores.shape[0] * frac):])
            LTE = labels[int(labels.shape[0] * frac):]
            scores, _, _ = logreg_wrapper(DTR_fin, LTR_fin, 0, pi=0.4, C_fn=1, C_fp=1, cal=True,
                                          DTR=vrow(DTR), LTR=LTR, DTE=DTE)
            return scores, LTE

    elif model == "SVM":
        DTR_fin, P = PCA(DTR_fin, PCA_SVM)
        DTE_fin = np.dot(P.T, DTE_fin)

        pol_kern = Kernel(d=2, K=K)
        svm = SVM(DTR_fin, LTR_fin, c, K, pol_kern.polynomial)
        scores = svm.scores(DTR_fin, LTR_fin, DTE_fin)

        if calibration == False:
            return scores, LTE_fin
        else:
            np.random.seed(seed)
            idx_new = np.random.permutation(scores.shape[0])
            scores = scores[idx_new]
            labels = LTE_fin[idx_new]

            frac = 7 / 10
            DTR = vrow(scores[:int(scores.shape[0] * frac)])
            LTR = labels[:int(labels.shape[0] * frac)]
            DTE = vrow(scores[int(scores.shape[0] * frac):])
            LTE = labels[int(labels.shape[0] * frac):]
            scores, _, _ = logreg_wrapper(DTR_fin, LTR_fin, 0, pi=0.4, C_fn=1, C_fp=1, cal=True,
                                          DTR=vrow(DTR), LTR=LTR, DTE=DTE)
            return scores, LTE

    elif model == 'GMM':
        scores = scores_gmm(DTR_fin, LTR_fin, DTE_fin, g_target, g_NONtarget)

        if calibration == False:
            return scores, LTE_fin
        else:
            np.random.seed(seed)
            idx_new = np.random.permutation(scores.shape[0])
            scores = scores[idx_new]
            labels = LTE_fin[idx_new]

            frac = 7 / 10
            DTR = vrow(scores[:int(scores.shape[0] * frac)])
            LTR = labels[:int(labels.shape[0] * frac)]
            DTE = vrow(scores[int(scores.shape[0] * frac):])
            LTE = labels[int(labels.shape[0] * frac):]
            scores, _, _ = logreg_wrapper(DTR_fin, LTR_fin, 0, pi=0.4, C_fn=1, C_fp=1, cal=True,
                                          DTR=vrow(DTR), LTR=LTR, DTE=DTE)
            return scores, LTE


fusion_output_file = '../FINALoutputs/fusion_models_EVAL.txt'
fusion_table = PrettyTable()
fusion_table.field_names = ['models', 'actDCF (pi = 0.1)', 'actDCF (pi = 0.5)',
                            'minDCF (pi = 0.1)', 'minDCF (pi = 0.5)', 'C_prim']

def best_scores_eval(DTR, LTR, DTE, LTE, seed=0, model='MVG'):

    if model == "MVG":
        DTR, P = PCA(DTR, PCA_MVG)
        DTE = np.dot(P.T, DTE)

        labels = LTE
        scores = scores_gaussian(DTR, LTR, DTE)
    elif model == "logreg":
        DTR, P = PCA(DTR, PCA_logreg)
        DTE = np.dot(P.T, DTE)

        labels = LTE
        scores = scores_logreg(DTR, LTR, DTE, lam)
    elif model == "SVM":
        DTR, P = PCA(DTR, PCA_SVM)
        DTE = np.dot(P.T, DTE)

        labels = LTE
        pol_kern = Kernel(d=2, K=K)
        svm = SVM(DTR, LTR, c, K, pol_kern.polynomial)
        scores = svm.scores(DTR, LTR, DTE)
    elif model == 'GMM':

        labels = LTE
        scores = scores_gmm(DTR, LTR, DTE, g_target, g_NONtarget)

    np.random.seed(seed)
    idx_new = np.random.permutation(scores.shape[0])
    scores = scores[idx_new]
    labels = labels[idx_new]

    frac = 7 / 10
    DTR = vrow(scores[:int(scores.shape[0] * frac)])
    LTR = labels[:int(labels.shape[0] * frac)]
    DTE = vrow(scores[int(scores.shape[0] * frac):])
    LTE = labels[int(labels.shape[0] * frac):]

    return DTR, LTR, DTE, LTE
def fusion_wrapper(DTR_fin, LTR_fin, DTE_fin, LTE_fin):
    original_stdout = sys.stdout

    with open(fusion_output_file, 'w') as f:
        sys.stdout = f
        models = np.array(['GMM', 'SVM', 'logreg', 'MVG'])
        DTR_gmm, LTR, DTE_gmm, LTE = best_scores_eval(DTR_fin, LTR_fin, DTE_fin, LTE_fin,model='GMM')
        DTR_svm, _, DTE_svm, _ = best_scores_eval(DTR_fin, LTR_fin, DTE_fin, LTE_fin, model='SVM')
        DTR_logreg, _, DTE_logreg, _ = best_scores_eval(DTR_fin, LTR_fin, DTE_fin, LTE_fin, model='logreg')
        DTR_mvg, _, DTE_mvg, _ = best_scores_eval(DTR_fin, LTR_fin, DTE_fin, LTE_fin, model='MVG')

        p = 0.5
        scores_fus_GMM, _, _ = logreg_wrapper(DTR_fin, LTR, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR_gmm), LTR=LTR,
                                              DTE=vrow(DTE_gmm))
        scores_fus_SVM, _, _ = logreg_wrapper(DTR_fin, LTR, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR_svm), LTR=LTR,
                                              DTE=vrow(DTE_svm))
        scores_fus_logreg, _, _ = logreg_wrapper(DTR_fin, LTR, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                                 DTR=vrow(DTR_logreg), LTR=LTR,
                                                 DTE=vrow(DTE_logreg))
        scores_fus_MVG, _, _ = logreg_wrapper(DTR_fin, LTR, 0, pi=p, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR_mvg), LTR=LTR,
                                              DTE=vrow(DTE_mvg))


        all_combins = [('SVM', 'MVG'), ('logreg', 'MVG'), ('SVM', 'logreg', 'MVG'), ('GMM', 'SVM', 'logreg', 'MVG')]

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


evaluation_test_file = '../FINALoutputs/evaluation_test.txt'
eval_test_tab = PrettyTable()
eval_test_tab.field_names = ['models', 'actDCF (pi = 0.1)', 'actDCF (pi = 0.5)',
                             'minDCF (pi = 0.1)', 'minDCF (pi = 0.5)', 'C_prim']
def evaluation_test(DTR_fin, DTE_fin, LTR_fin, LTE_fin):
    original_stdout = sys.stdout

    with open(evaluation_test_file, 'w') as f:
        sys.stdout = f
        models = ['GMM', 'SVM', 'logreg', 'MVG']
        for model in models:
            scores, LTE = scores_test(DTR_fin, DTE_fin, LTR_fin, LTE_fin, model=model, calibration=False)
            minDCF0 = evaluation.minDCF(scores, LTE, 0.1, 1, 1)
            actDCF0 = evaluation.Bayes_risk_normalized(scores, LTE, 0.1, 1, 1)
            minDCF1 = evaluation.minDCF(scores, LTE, 0.5, 1, 1)
            actDCF1 = evaluation.Bayes_risk_normalized(scores, LTE, 0.5, 1, 1)
            C_prim = np.mean([minDCF0, minDCF1])
            eval_test_tab.add_row([model, actDCF0, actDCF1, minDCF0, minDCF1, C_prim])

        print(eval_test_tab)
        sys.stdout = original_stdout

def DET_plot_evaluation(DTR_fin, DTE_fin, LTR_fin, LTE_fin):
    models = np.array(['GMM', 'SVM', 'logreg', 'MVG'])
    colors = ['red', 'green', 'blue', 'fuchsia']

    plt.figure()
    for i, model in enumerate(models):
        if model == 'GMM':
            scores, labels = scores_test(DTR_fin, DTE_fin, LTR_fin, LTE_fin, model=model, calibration=False)
        else:
            scores, labels = scores_test(DTR_fin, DTE_fin, LTR_fin, LTE_fin, model=model, calibration=True)
        evaluation.DET_curve(scores, labels, model=model, color=colors[i])
        if i == 3:
            plt.savefig('../Plots/DET_plot_EVALUATIONmodels.png', bbox_inches='tight')
            # plt.show(bbox_inches='tight')
            plt.close()

def Bayes_plot_evaluation(DTR_fin, DTE_fin, LTR_fin, LTE_fin):
    models = np.array(['GMM', 'SVM', 'logreg', 'MVG'])
    colors = ['red', 'green', 'blue', 'fuchsia']

    plt.figure()
    for i, model in enumerate(models):
        scores, labels = scores_test(DTR_fin, DTE_fin, LTR_fin, LTE_fin, model=model)
        evaluation.Bayes_error_plots(scores, labels, model=model, color=colors[i])
        if i == 3:
            plt.savefig('../Plots/Bayes_plot_EVALUATIONmodels.png', bbox_inches='tight')
            # plt.show(bbox_inches='tight')
            plt.close()


####################################### OPTIMAL CHOICES EVALUATION ##############################################################################

def optimal_wrap_evaluation(DTR_fin, DTE_fin, LTR_fin, LTE_fin, model='MVG', seed=0):

    plt.figure()
    C_prim_list = []
    minDCF_list = []
    PCA_list = ['No PCA', 9, 8, 7, 6, 5]
    lambda_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    g_list = [1,2,4,8,16]
    c_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    K_list = [0.1, 1, 10]

    if model == "MVG":
        for m in PCA_list:
            if m != 'No PCA':
                DTR, P = PCA(DTR_fin, m)
                DTE = np.dot(P.T, DTE_fin)
            else:
                DTR = DTR_fin
                DTE = DTE_fin

            scores = scores_gaussian(DTR, LTR_fin, DTE)
            labels = LTE_fin
            minDCF0 = evaluation.minDCF(scores,labels,0.1,1,1)
            minDCF1 = evaluation.minDCF(scores, labels, 0.5, 1, 1)
            C_prim_list.append(np.mean([minDCF0, minDCF1]))
            minDCF_list.append(minDCF0)

        print(f'm min: {PCA_list[np.argmin(C_prim_list)]}, C_prim = {np.min(C_prim_list)}')
        print(f'm min: {PCA_list[np.argmin(minDCF_list)]}, minDCF optimal = {np.min(minDCF_list)}')
        plt.plot(PCA_list, C_prim_list)
        plt.xlabel('\\textbf{PCA}', fontsize=16)
        plt.ylabel('\\textbf{$C_{prim}$}', fontsize=16)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.grid()
        # plt.show()
        plt.savefig('../Plots/EVAL_optimal_MVG.png', bbox_inches='tight')
        plt.close()

    elif model == "logreg":
        for lam in lambda_list:
            DTR, P = PCA(DTR_fin, PCA_logreg)
            DTE = np.dot(P.T, DTE_fin)

            scores = scores_logreg(DTR, LTR_fin, DTE, lam)

            np.random.seed(seed)
            idx_new = np.random.permutation(scores.shape[0])
            scores = scores[idx_new]
            labels = LTE_fin[idx_new]

            frac = 7 / 10
            DTR = vrow(scores[:int(scores.shape[0] * frac)])
            LTR = labels[:int(labels.shape[0] * frac)]
            DTE = vrow(scores[int(scores.shape[0] * frac):])
            LTE = labels[int(labels.shape[0] * frac):]
            scores, _, _ = logreg_wrapper(DTR_fin, LTR_fin, 0, pi=0.4, C_fn=1, C_fp=1, cal=True,
                                          DTR=vrow(DTR), LTR=LTR, DTE=DTE)
            labels = LTE

            minDCF0 = evaluation.minDCF(scores,labels,0.1,1,1)
            minDCF1 = evaluation.minDCF(scores, labels, 0.5, 1, 1)
            C_prim_list.append(np.mean([minDCF0, minDCF1]))
            minDCF_list.append(minDCF0)

        print(f'lambda min: {lambda_list[np.argmin(C_prim_list)]}, C_prim = {np.min(C_prim_list)}')
        print(f'lambda min: {lambda_list[np.argmin(minDCF_list)]}, minDCF optimal = {np.min(minDCF_list)}')
        plt.plot(lambda_list, C_prim_list)
        plt.xlabel('\\textbf{$\lambda$}', fontsize=16)
        plt.ylabel('\\textbf{$C_{prim}$}', fontsize=16)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.xscale('log')
        ax = plt.gca()
        ax.set_xticks(lambda_list)
        plt.grid()
        # plt.show()
        plt.savefig('../Plots/EVAL_optimal_logreg.png', bbox_inches='tight')
        plt.close()

    elif model == "GMM":
        for g_target in [1,2,4]:
            C_prim_list = []
            minDCF_list = []
            for g_NONtarget in g_list:
                scores = scores_gmm(DTR_fin, LTR_fin, DTE_fin, g_target, g_NONtarget)
                labels =  LTE_fin
                minDCF0 = evaluation.minDCF(scores, labels, 0.1, 1, 1)
                minDCF1 = evaluation.minDCF(scores, labels, 0.5, 1, 1)
                C_prim_list.append(np.mean([minDCF0, minDCF1]))
                minDCF_list.append(minDCF0)

            print(f'g_target: {g_target}')
            print(f'g_NONtarget min: {g_list[np.argmin(C_prim_list)]}, C_prim = {np.min(C_prim_list)}')
            print(f'g_NONtarget min: {g_list[np.argmin(minDCF_list)]}, minDCF optimal = {np.min(minDCF_list)}')
            plt.plot(np.arange(len(g_list)), C_prim_list, label='$K_{target}$: '+f'{g_target}')
            plt.legend(fontsize=16)
            plt.xlabel('\\textbf{$K_{non target}$}', fontsize=16)
            plt.ylabel('\\textbf{$C_{prim}$}', fontsize=16)
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)
            ax = plt.gca()
            ax.set_xticks(np.arange(len(g_list)))
            ax.set_xticklabels(g_list)
            plt.grid()
        # plt.show()
        plt.savefig('../Plots/EVAL_optimal_gmm.png', bbox_inches='tight')
        plt.close()

    elif model == "GMM calibrated":
        for g_target in [1,2,4]:
            C_prim_list = []
            for g_NONtarget in g_list:
                scores = scores_gmm(DTR_fin, LTR_fin, DTE_fin, g_target, g_NONtarget)
                np.random.seed(seed)
                idx_new = np.random.permutation(scores.shape[0])
                scores = scores[idx_new]
                labels = LTE_fin[idx_new]

                frac = 7 / 10
                DTR = vrow(scores[:int(scores.shape[0] * frac)])
                LTR = labels[:int(labels.shape[0] * frac)]
                DTE = vrow(scores[int(scores.shape[0] * frac):])
                LTE = labels[int(labels.shape[0] * frac):]
                scores, _, _ = logreg_wrapper(DTR_fin, LTR_fin, 0, pi=0.4, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR), LTR=LTR, DTE=DTE)
                labels = LTE
                minDCF0 = evaluation.minDCF(scores, labels, 0.1, 1, 1)
                minDCF1 = evaluation.minDCF(scores, labels, 0.5, 1, 1)
                C_prim_list.append(np.mean([minDCF0, minDCF1]))

            print(f'g_target: {g_target}')
            print(f'g_NONtarget min: {g_list[np.argmin(C_prim_list)]}, C_prim = {np.min(C_prim_list)}')
            plt.plot(np.arange(len(g_list)), C_prim_list, label='$K_{target}$: '+f'{g_target}')
            plt.legend(fontsize=16)
            plt.xlabel('\\textbf{$K_{non target}$}', fontsize=16)
            plt.ylabel('\\textbf{$C_{prim}$}', fontsize=16)
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)
            ax = plt.gca()
            ax.set_xticks(np.arange(len(g_list)))
            ax.set_xticklabels(g_list)
            plt.grid()
        # plt.show()
        plt.savefig('../Plots/EVAL_optimal_gmm_calibrated.png', bbox_inches='tight')
        plt.close()

    elif model == "SVM":
        for K in K_list:
            C_prim_list = []
            minDCF_list = []
            for c in c_list:
                DTR_fin, P = PCA(DTR_fin, PCA_SVM)
                DTE_fin = np.dot(P.T, DTE_fin)

                pol_kern = Kernel(d=2, K=K)
                svm = SVM(DTR_fin, LTR_fin, c, K, pol_kern.polynomial)
                scores = svm.scores(DTR_fin, LTR_fin, DTE_fin)

                np.random.seed(seed)
                idx_new = np.random.permutation(scores.shape[0])
                scores = scores[idx_new]
                labels = LTE_fin[idx_new]

                frac = 7 / 10
                DTR = vrow(scores[:int(scores.shape[0] * frac)])
                LTR = labels[:int(labels.shape[0] * frac)]
                DTE = vrow(scores[int(scores.shape[0] * frac):])
                LTE = labels[int(labels.shape[0] * frac):]
                scores, _, _ = logreg_wrapper(DTR_fin, LTR_fin, 0, pi=0.4, C_fn=1, C_fp=1, cal=True,
                                              DTR=vrow(DTR), LTR=LTR, DTE=DTE)
                labels = LTE

                minDCF0 = evaluation.minDCF(scores,labels,0.1,1,1)
                minDCF1 = evaluation.minDCF(scores, labels, 0.5, 1, 1)
                C_prim_list.append(np.mean([minDCF0, minDCF1]))
                minDCF_list.append(minDCF0)

            print(f'K: {K}')
            print(f'c min: {c_list[np.argmin(C_prim_list)]}, C_prim = {np.min(C_prim_list)}')
            print(f'c min: {c_list[np.argmin(minDCF_list)]}, minDCF optimal = {np.min(minDCF_list)}')
            plt.plot(c_list, C_prim_list, label='\\textbf{K = ' +f'{K}'+'}')
            plt.xlabel('\\textbf{$c$}', fontsize=16)
            plt.ylabel('\\textbf{$C_{prim}$}', fontsize=16)
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)
            plt.legend(fontsize=14)
            plt.xscale('log')
            ax = plt.gca()
            ax.set_xticks(c_list)
            plt.grid()
        # plt.show()
        plt.savefig('../Plots/EVAL_optimal_SVM.png', bbox_inches='tight')
        plt.close()