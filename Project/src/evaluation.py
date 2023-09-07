import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(pred, f_labels):
    confusion_matrix = np.zeros((2,2))
    for ii in range(len(pred)):
        confusion_matrix[pred[ii]][f_labels[ii]] += 1
    return confusion_matrix

def Bayes_risk(p, C_fn, C_fp, pred, f_labels):
    M = np.zeros((2, 2))
    for ii in range(len(pred)):
        M[pred[ii]][f_labels[ii]] += 1
    FNR = M[0][1] / (M[0][1] + M[1][1])
    FPR = M[1][0] / (M[0][0] + M[1][0])
    DCF = p*C_fn*FNR + (1-p)*C_fp*FPR
    return DCF

def Bayes_risk_normalized(f_llr,f_labels,p,C_fn,C_fp):
    threshold = np.log((p * C_fn) / ((1 - p) * C_fp))
    pred = np.where(f_llr > - threshold,1,0)
    M = np.zeros((2, 2))
    for ii in range(len(pred)):
        M[int(pred[ii])][int(f_labels[ii])] += 1
    FNR = M[0][1] / (M[0][1] + M[1][1])
    FPR = M[1][0] / (M[0][0] + M[1][0])
    DCF = p * C_fn * FNR + (1 - p) * C_fp * FPR
    DCF_dummy = min(p*C_fn,(1-p)*C_fp)
    return DCF / DCF_dummy

def minDCF(f_llr,f_labels,p,C_fn,C_fp):
    t_set = np.sort(np.append(f_llr,[np.inf,-np.inf]))
    DCF_min = []
    for threshold in t_set:
        pred = np.where(f_llr > threshold,1,0)
        M = np.zeros((2, 2))
        for ii in range(len(pred)):
            M[int(pred[ii])][int(f_labels[ii])] += 1
        FNR = M[0][1] / (M[0][1] + M[1][1])
        FPR = M[1][0] / (M[0][0] + M[1][0])
        DCF = p * C_fn * FNR + (1 - p) * C_fp * FPR
        DCF_dummy = min(p*C_fn,(1-p)*C_fp)
        DCF_min.append(DCF / DCF_dummy)
    return np.min(DCF_min)

def ROC_curve(f_llr,f_labels):
    t_set = np.sort(np.append(f_llr, [np.inf, -np.inf]))
    FPR = []
    TPR = []
    for aa in t_set:
        threshold = aa
        pred = np.where(f_llr > - threshold, 1, 0)
        M = confusion_matrix(pred, f_labels)
        TPR.append( 1- (M[0][1] / (M[0][1] + M[1][1])))
        FPR.append(M[1][0] / (M[0][0] + M[1][0]))
    plt.figure()
    plt.plot(FPR,TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0,1])
    plt.xlim([0,1])
    return plt.show()

def DET_curve(f_llr,f_labels):
    t_set = np.sort(np.append(f_llr, [np.inf, -np.inf]))
    FNR = []
    FPR = []
    for aa in t_set:
        threshold = aa
        pred = np.where(f_llr > - threshold, 1, 0)
        M = confusion_matrix(pred, f_labels)
        FNR.append(M[0][1] / (M[0][1] + M[1][1]))
        FPR.append(M[1][0] / (M[0][0] + M[1][0]))
    plt.figure()
    plt.plot(FPR, FNR)
    plt.xlabel('FPR')
    plt.ylabel('FNR')
    return plt.show()


def Bayes_error_plots(llr,labels):
    plt.figure()
    try:
        for i in range(len(llr)):
            DCF_min = []
            DCF = []
            effPriorLogOdds = np.linspace(-3, 3, 25)
            for p_tilde in effPriorLogOdds:
                prior = 1 / (1 + np.exp(-p_tilde))
                DCF.append(Bayes_risk_normalized(llr[i],labels[i],prior,1,1))
                DCF_min.append(minDCF(llr[i],labels[i],prior,1,1))
            plt.plot(effPriorLogOdds, DCF, label= 'DCF' if i == 0 else 'DCF_eps1', color ='r' if i == 0 else 'y')
            plt.plot(effPriorLogOdds, DCF_min, label='min DCF' if i == 0 else 'min DCF_eps1', color ='b' if i == 0 else 'c')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
    except: TypeError
    DCF_min = []
    DCF = []
    effPriorLogOdds = np.linspace(-3, 3, 25)
    for p_tilde in effPriorLogOdds:
        prior = 1 / (1 + np.exp(-p_tilde))
        DCF.append(Bayes_risk_normalized(llr, labels, prior, 1, 1))
        DCF_min.append(minDCF(llr, labels, prior, 1, 1))
    plt.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    plt.plot(effPriorLogOdds, DCF_min, label='min DCF', color='b')
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.grid()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    return plt.show()