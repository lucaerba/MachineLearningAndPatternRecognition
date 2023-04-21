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
    N = x.shape[1]
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

D, L = load_iris()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

mu_ML = {}
C_ML = {}
D_class = {}

for i in range(3):
    D_class[i] = DTR[:, LTR == i]
    mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class[i])

N_c = np.array([v.shape[1] for k, v in D_class.items()])
sigma = [v for k, v in C_ML.items()]
sigma_star = float(np.sum(N_c))**-1 * np.dot(vrow(N_c),np.reshape(sigma, (3,16)))
sigma_star = np.reshape(sigma_star, (4,4))
S = []
P_C = 1/3

for i in range(3):
    D_class = DTE
    like = vrow(np.array(np.exp(logpdf_GAU_ND(D_class, mu_ML[i], sigma_star))))
    S.extend(like)

SJoint = np.array(S) * P_C
# Corr_Sol = np.load('../../SJoint_MVG.npy')

SMarginal = vrow(SJoint.sum(0))

SPost = SJoint / SMarginal

predicted_labels = np.argmax(SPost, axis=0)

check = predicted_labels == LTE
# print(check)

acc = len(check[check == True]) / len(LTE)
print(1-acc)











