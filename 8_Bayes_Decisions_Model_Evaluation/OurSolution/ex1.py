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

def logpdf_GAU_ND(X, mu, C):
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
    P_C = 1/3

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

if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    Sjoint = score_matrix_TiedMVG(DTR, LTR, DTE)
    pred, acc = predicted_labels_and_accuracy(Sjoint,LTE)

    confusion_matrix = np.zeros((3,3))
    for ii in range(len(pred)):
        confusion_matrix[pred[ii]][LTE[ii]] += 1
    print(confusion_matrix)