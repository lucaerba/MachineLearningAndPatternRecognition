import scipy as sp
import numpy as np    
import threading
import evaluation

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v
############################## MVG and NAIVE + TIED ###################################
        
def logpdf_GAU_ND(X, mu, C):
    XC = X - vcol(mu)
    M = X.shape[0]
    const = - 0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC*np.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v

def mu_and_sigma_ML(x):
    N = x.shape[1]

    mu_ML = np.mean(x,axis=1)
    x_cent = x - mu_ML[:, np.newaxis]

    sigma_ML = 1/N * np.dot(x_cent,x_cent.T)

    return mu_ML, sigma_ML

def loglikelihood(x, mu_ML, C_ML):
    l = np.sum(logpdf_GAU_ND(x, mu_ML, C_ML))
    return l

prior_prob = np.log(np.array([[0.1],[0.9]]))
N_classes = 2

def score_matrix_MVG(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    S = []

    for i in range(N_classes):
        D_class = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class)

    for i in range(N_classes):
        D_class = DTE
        like = vrow(np.array((logpdf_GAU_ND(D_class, mu_ML[i], C_ML[i]))))
        S.extend(like)
    SJoint = np.array(S) + prior_prob
    return SJoint

def score_matrix_NaiveBayes(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    S = []

    for i in range(N_classes):
        D_class = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class)
        C_ML[i] = C_ML[i] * np.eye(len(C_ML[i]))

    for i in range(N_classes):
        D_class = DTE
        like = vrow(np.array((logpdf_GAU_ND(D_class, mu_ML[i], C_ML[i]))))
        S.extend(like)

    Sjoint = np.array(S) + prior_prob
    return Sjoint

def score_matrix_TiedMVG(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    D_class = {}

    for i in range(N_classes):
        D_class[i] = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class[i])

    N_c = np.array([v.shape[1] for k, v in D_class.items()])
    sigma = [v for k, v in C_ML.items()]
    sigma_star = float(np.sum(N_c)) ** -1 * np.dot(vrow(N_c), np.reshape(sigma, (N_classes, DTR.shape[0]**2)))
    sigma_star = np.reshape(sigma_star, (DTR.shape[0], DTR.shape[0]))
    S = []

    for i in range(N_classes):
        D_class = DTE
        like = vrow(np.array((logpdf_GAU_ND(D_class, mu_ML[i], sigma_star))))
        S.extend(like)

    SJoint = np.array(S) + prior_prob
    return SJoint

def score_matrix_TiedNaiveBayes(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    D_class = {}

    for i in range(N_classes):
        D_class[i] = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class[i])
        C_ML[i] = C_ML[i] * np.eye(len(C_ML[i]))

    N_c = np.array([v.shape[1] for k, v in D_class.items()])
    sigma = [v for k, v in C_ML.items()]
    sigma_star = float(np.sum(N_c)) ** -1 * np.dot(vrow(N_c), np.reshape(sigma, (N_classes, DTR.shape[0]**2)))
    sigma_star = np.reshape(sigma_star, (DTR.shape[0], DTR.shape[0]))
    S = []

    for i in range(N_classes):
        D_class = DTE
        like = vrow(np.array((logpdf_GAU_ND(D_class, mu_ML[i], sigma_star))))
        S.extend(like)

    SJoint = np.array(S) + prior_prob
    return SJoint

def predicted_labels_and_accuracy(S, LTE):
    SMarginal = vrow(sp.special.logsumexp(S, axis=0))
    SPost = np.exp(S - SMarginal)
    predicted_labels = np.argmax(SPost, axis=0)
    check = predicted_labels == LTE
    acc = len(check[check == True]) / len(LTE)
    return predicted_labels, acc

def Kfold_cross_validation(D, L, K, seed=1, func=score_matrix_MVG):
    nSamp = int(D.shape[1]/K)
    residuals = D.shape[1] - nSamp*K
    sub_arr = np.ones((K, 1)) * nSamp
    
    if residuals != 0:
        sub_arr = np.array([int(x+1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    
    err = []
    pred = []
    minDCF = []
    for i in range(K):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i+1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        Sjoint = func(DTR, LTR, DTE)
        pred_i, acc_i = predicted_labels_and_accuracy(Sjoint, LTE)
        pred.append(pred_i)
        err.append(1-acc_i)

        scores = - Sjoint[0,:] + Sjoint[1,:]

        minDCF.append(evaluation.minDCF(scores, LTE, 0.1, 1, 1))

    return np.min(err),pred[np.argmin(err)],np.mean(minDCF)

