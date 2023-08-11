import numpy as np
import matplotlib.pyplot as plt


"""
This file contains useful functions and models implemented in laboratories (each model returns values useful to compute accuracy)
"""

# number of classes in dataset
Nc = 2


def vrow(x):
    return x.reshape((1, x.size))


def vcol(m):
    """Reshape as a column vector input m"""
    return m.reshape((m.size, 1))


def loadData(filename):
    """
    Load dataset samples and relative labels from file. Returns D data-matrix where each column is a sample and L containing labels.
    The i-th sample of D is D[:,i] and the relative label is L[i].
    """
    D = []
    L = []
    with open(filename, 'r') as f:
        for line in f:
            sample = np.array([float(i) for i in line.split(",")]
                              [0:10]).reshape(10, 1)  # 6 features
            label = line.split(",")[10].rstrip()
            D.append(sample)
            L.append(label)
        D = np.hstack(D)
        L = np.array(L, dtype=np.int32)
        return D, L


def plot_scatter(D, L, title):
    # make data that will be plotted
    ita = (L == 1)
    no_ita = (L == 0)

    data_ita = D[:, ita]
    data_no_ita = D[:, no_ita]

    plt.figure()
    plt.scatter(data_ita[0, :], data_ita[1, :], label='Italian', s=16)
    plt.scatter(data_no_ita[0, :], data_no_ita[1, :],label='Not Italian', s=16)

    plt.title(title)
    plt.legend()
    plt.show()

# m is number of dimension to keep after PCA


def PCA(D, m):
    N = D.shape[1]
    # compute mean by column of the dataset for each dimension, note that mu is a row vector of shape (4,)
    mu = vcol(D.mean(1))
    DC = D - mu  # center data
    C = np.dot(DC, DC.T)/N  # compute the covariance matrix of centered data
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]  # get 0:m highest
    DP = np.dot(P.T, D)
    return DP, P

def apply_Z_Norm(DTR, DTE):
    """
    Data after Z-Norm = (x – μ) / σ where:
    x: Original value
    μ: Mean of training data 
    σ: Standard deviation of training data
    Z-norm will be applied also on test data with the same parameters computed from training data
    Returns DTR and DTE with Z-Norm applied
    """
    mu = vcol(DTR.mean(1))
    std = vcol(DTR.std(1))

    DTR_znorm = (DTR - mu) / std
    DTE_znorm = (DTE - mu) / std
    return DTR_znorm, DTE_znorm

def logpdf_GAU_ND(x, mu, C):
    Y = []
    for i in range(x.shape[1]):
        Y.append(logpdf_GAU_ND2(x[:, i:i+1], mu, C))
    return np.array(Y).ravel()


def logpdf_GAU_ND2(x, mu, C):
    xc = x - mu
    M = x.shape[0]
    const = - 0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = np.dot(xc.T, np.dot(L, xc)).ravel()
    return const - 0.5 * logdet - 0.5 * v
