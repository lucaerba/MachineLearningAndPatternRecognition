import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np

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

mu = {}
C = {}
mu_ML = {}
C_ML = {}
for i in range(3):
    D_class = D[:, L == i]
    n_c = len(D_class[0])

    mu[i] = np.array(np.mean(D_class, 1))
    DC_class = D_class - vcol(mu[i])
    C[i] = n_c**-1 * np.dot(DC_class,DC_class.T)

    mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class)

    # print(np.max(np.abs(vcol(mu[i]) - mu_ML[i])), np.max(np.abs(C[i] - C_ML[i])))

# print(mu, '\n\n', mu_ML)
# print(C, '\n\n', C_ML)

# print(mu, '\n\n', C)


S = []
P_C = 1/3

for i in range(3):
    D_class = D[:, L == i]

    like = np.array(np.exp((logpdf_GAU_ND(D_class, mu[i], C[i]))))
    S.append(like)

SJoint = np.reshape((np.array(S) * P_C), (3,50))
Corr_Sol = np.load('SJoint_MVG.npy')
print(SJoint)
print(Corr_Sol)
# err = (np.abs(Corr_Sol - SJoint))
# print("error matrix: ", err)



