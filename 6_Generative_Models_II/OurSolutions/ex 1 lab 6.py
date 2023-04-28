import sklearn.datasets
import numpy as np
import scipy as sp
import sys

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def load(file):
    attributes = []
    fam_list = []
    families = {
        'Iris-setosa' : 0,
        'Iris-versicolor' : 1,
        'Iris-virginica' : 2
    }

    with open(file, 'r') as file:
        for line in file:
            vector = line.split(',')
            flower_fam = vector.pop(-1).strip()
            fam_ind = families[flower_fam]
            vector = np.array([float(i) for i in vector]).reshape(len(vector),1)
            attributes.append(vector)
            fam_list.append(fam_ind)

    return np.hstack(attributes), np.array((fam_list))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def logpdf_GAU_ND(x, mu, C):
    logN = []
    x = np.array(x)
    M = x.shape[0]
    if len(x.shape) == 2:
        N = x.shape[1]
    else:
        N = 1
        x = np.reshape(x,(4,1))
    C_inv = np.linalg.inv(C)
    xc = (x - vcol(mu))
    x_cent = np.reshape(xc, (M*N, 1), order='F')
    i = 0
    while i in range(N*M):
        xx = x_cent[i:i+M]
        first_term = -.5* M*np.log(2*np.pi)
        second_term = -.5* np.linalg.slogdet(C)[1]
        third_term = -.5* np.dot(vrow(xx),np.ones((M,1))*np.dot(C_inv,vcol(xx)))
        i += M
        logN.append(first_term + second_term + third_term)
    return np.vstack(logN)

def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC*np.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v

def mu_and_sigma_ML(x):
    N = x.shape[1]
    M = x.shape[0]

    mu_ML = []
    sigma_ML = []

    for i in range(M):
        mu_ML.append(np.sum(x[i,:]) / N)

    x_cent = x - np.reshape(mu_ML, (M,1))
    for i in range(M):
        for j in range(M):
            sigma_ML.append(np.dot(x_cent[i,:],x_cent[j,:].T) / N)

    return np.vstack(mu_ML), np.reshape(sigma_ML, (M,M))

def loglikelihood(x, mu_ML, C_ML):
    l = np.sum(logpdf_GAU_ND(x, mu_ML, C_ML))
    return l

def score_matrix_MVG(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    S = []
    P_C = 1 / 3

    for i in range(3):
        D_class = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class)

    for i in range(3):
        D_class = DTE
        like = vrow(np.array(np.exp(logpdf_GAU_ND(D_class, mu_ML[i], C_ML[i]))))
        S.extend(like)
    SJoint = np.array(S) * P_C
    return SJoint

def score_matrix_NaiveBayes(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    S = []
    P_C = 1 / 3

    for i in range(3):
        D_class = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class)
        C_ML[i] = C_ML[i] * np.eye(len(C_ML[i]))

    for i in range(3):
        D_class = DTE
        like = vrow(np.array(np.exp(logpdf_GAU_ND(D_class, mu_ML[i], C_ML[i]))))
        S.extend(like)

    Sjoint = np.array(S) * P_C
    return Sjoint

def score_matrix_TiedMVG(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    D_class = {}

    for i in range(3):
        D_class[i] = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class[i])

    N_c = np.array([v.shape[1] for k, v in D_class.items()])
    sigma = [v for k, v in C_ML.items()]
    sigma_star = float(np.sum(N_c)) ** -1 * np.dot(vrow(N_c), np.reshape(sigma, (3, 16)))
    sigma_star = np.reshape(sigma_star, (4, 4))
    S = []
    P_C = 1 / 3

    for i in range(3):
        D_class = DTE
        like = vrow(np.array(np.exp(logpdf_GAU_ND(D_class, mu_ML[i], sigma_star))))
        S.extend(like)

    SJoint = np.array(S) * P_C
    return SJoint

def score_matrix_TiedNaiveBayes(DTR, LTR, DTE):
    mu_ML = {}
    C_ML = {}
    D_class = {}

    for i in range(3):
        D_class[i] = DTR[:, LTR == i]
        mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class[i])
        C_ML[i] = C_ML[i] * np.eye(len(C_ML[i]))

    N_c = np.array([v.shape[1] for k, v in D_class.items()])
    sigma = [v for k, v in C_ML.items()]
    sigma_star = float(np.sum(N_c)) ** -1 * np.dot(vrow(N_c), np.reshape(sigma, (3, 16)))
    sigma_star = np.reshape(sigma_star, (4, 4))
    S = []
    P_C = 1 / 3

    for i in range(3):
        D_class = DTE
        like = vrow(np.array(np.exp(logpdf_GAU_ND(D_class, mu_ML[i], sigma_star))))
        S.extend(like)

    SJoint = np.array(S) * P_C
    return SJoint

def predicted_labels_and_accuracy(S, LTE):
    SMarginal = vrow(S.sum(0))
    SPost = S / SMarginal
    predicted_labels = np.argmax(SPost, axis=0)
    check = predicted_labels == LTE
    acc = len(check[check == True]) / len(LTE)
    return predicted_labels, acc

def LOO_cross_validation(D, L, seed=1):
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    err = []
    for i in range(D.shape[1]):
        idxTest = idx[i]
        idxTrain = idx[idx != idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = np.ones((1,1)) * L[idxTest]
        Sjoint = score_matrix_MVG(DTR, LTR, DTE)
        pred, acc_i = predicted_labels_and_accuracy(Sjoint, LTE)
        err.append(1-acc_i)
    return np.mean(err)

def Kfold_cross_validation(D, L, K, seed=1):
    nSamp = int(D.shape[1]/K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    err = []
    for i in range(K):
        idxTest = idx[i*nSamp:nSamp*(i+1)]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        Sjoint = score_matrix_MVG(DTR, LTR, DTE)
        pred, acc_i = predicted_labels_and_accuracy(Sjoint, LTE)
        err.append(1-acc_i)
    return np.mean(err)


D, L = load_iris()
# (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
#
# Sjoint = score_matrix_TiedNaiveBayes(DTR, LTR, DTE)
#
# pred, acc = predicted_labels_and_accuracy(Sjoint,LTE)

# err = LOO_cross_validation(D, L)
err = Kfold_cross_validation(D, L, 5)
print(err)





















### MVG with log-densities ###
# S = []
# for i in range(3):
#     D_class = DTE
#     like = vrow(np.array(logpdf_GAU_ND(D_class, mu_ML[i], C_ML[i])))
#     S.extend(like)
#
# SJoint_log = np.array(S) + np.log(P_C)
# SMarginal_log = vrow(sp.special.logsumexp(SJoint_log, axis=0))
# logSPost = SJoint_log - SMarginal_log
# SPost = np.exp(logSPost)
#
# predicted_labels = np.argmax(SPost, axis=0)
#
# check = predicted_labels == LTE
# # print(check)
#
# acc = len(check[check == True]) / len(LTE)


