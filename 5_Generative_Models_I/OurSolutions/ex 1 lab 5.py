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

input_file = sys.argv[1]
D, L = load(input_file)
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

mu_ML = {}
C_ML = {}

for i in range(3):
    D_class = DTR[:, LTR == i]
    mu_ML[i], C_ML[i] = mu_and_sigma_ML(D_class)

# print(mu_ML, '\n\n', C_ML)

S = []
P_C = 1/3

for i in range(3):
    D_class = DTE
    like = vrow(np.array(np.exp(logpdf_GAU_ND(D_class, mu_ML[i], C_ML[i]))))
    S.extend(like)

# np.reshape(S, (3,50))

# print(S)
SJoint = np.array(S) * P_C
Corr_Sol = np.load('../../SJoint_MVG.npy')

# print(np.max(err))
# print(SJoint)
# print(Corr_Sol)
# err = (np.abs(Corr_Sol - SJoint))
# print("error: ", np.max(err))

SMarginal = vrow(SJoint.sum(0))

SPost = SJoint / SMarginal

predicted_labels = np.argmax(SPost, axis=0)

check = predicted_labels == LTE
# print(check)

acc = len(check[check == True]) / len(LTE)

S = []
for i in range(3):
    D_class = DTE
    like = vrow(np.array(logpdf_GAU_ND(D_class, mu_ML[i], C_ML[i])))
    S.extend(like)

SJoint_log = np.array(S) + np.log(P_C)
SMarginal_log = vrow(sp.special.logsumexp(SJoint_log, axis=0))
logSPost = SJoint_log - SMarginal_log
SPost = np.exp(logSPost)

predicted_labels = np.argmax(SPost, axis=0)

check = predicted_labels == LTE
# print(check)

acc = len(check[check == True]) / len(LTE)


