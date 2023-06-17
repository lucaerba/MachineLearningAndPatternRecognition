import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from GMM_load import load_gmm
import sklearn.datasets

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    C_inv = np.linalg.inv(C)
    x_c = x - mu
    first_term = -.5* M*np.log(2*np.pi)
    second_term = -.5* np.linalg.slogdet(C)[1]
    third_term = -.5* (x_c*np.dot(C_inv,x_c)).sum(0)
    logN = first_term + second_term + third_term
    return np.hstack(logN)

def mu_and_sigma_ML(x):
    N = x.shape[1]

    mu_ML = np.mean(x,axis=1)
    x_cent = x - mu_ML[:, np.newaxis]

    sigma_ML = 1/N * np.dot(x_cent,x_cent.T)

    return mu_ML, sigma_ML

def loglikelihood(x, mu_ML, C_ML):
    l = np.sum(logpdf_GAU_ND(x, mu_ML, C_ML))
    return l

def logpdf_GMM(X, gmm):
    S = np.zeros((len(gmm), X.shape[1]))
    for g in range(len(gmm)):
        (w,mu,C) = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, vcol(mu), C) + np.log(w)
    logdens = sp.special.logsumexp(S, axis=0)
    return S, np.hstack(logdens)

def GMM_EM(X,gmm,psi=0.01):
    err = 1E-6
    log_l_OLD = 0
    diff = np.inf
    N = X.shape[1]

    while diff > err:
        gmm_NEW = []
        S, logdens = logpdf_GMM(X, gmm)
        Post = np.exp(S - logdens)
        for g in range(len(gmm)):
            gamma = Post[g,:]
            Z_g = np.sum(gamma)
            F_g = np.dot(gamma,X.T)
            S_g = np.dot(X, (vrow(gamma)*X).T)
            w = Z_g/N
            mu = vcol(F_g/Z_g)
            Sigma = S_g/Z_g - np.dot(mu,mu.T)
            U, s, _ = np.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = np.dot(U, vcol(s) * U.T)

            gmm_NEW.append((w,mu,Sigma))

        log_l_NEW = logdens.sum()/N
        # print(log_l_NEW)
        diff = np.abs(log_l_NEW - log_l_OLD)

        log_l_OLD = log_l_NEW
        gmm = gmm_NEW

    return gmm

def GMM_EM_diag(X,gmm,psi=0.01):
    err = 1E-6
    log_l_OLD = 0
    diff = np.inf
    N = X.shape[1]

    while diff > err:
        gmm_NEW = []
        S, logdens = logpdf_GMM(X, gmm)
        Post = np.exp(S - logdens)
        for g in range(len(gmm)):
            gamma = Post[g, :]
            Z_g = np.sum(gamma)
            F_g = np.dot(gamma, X.T)
            S_g = np.dot(X, (vrow(gamma) * X).T)
            w = Z_g / N
            mu = vcol(F_g / Z_g)
            Sigma = S_g / Z_g - np.dot(mu, mu.T)
            Sigma = Sigma * np.eye(Sigma.shape[0])
            U, s, _ = np.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = np.dot(U, vcol(s) * U.T)

            gmm_NEW.append((w, mu, Sigma))

        log_l_NEW = logdens.sum() / N
        # print(log_l_NEW)
        diff = np.abs(log_l_NEW - log_l_OLD)

        log_l_OLD = log_l_NEW
        gmm = gmm_NEW

    return gmm

def GMM_EM_tied(X,gmm,psi=0.01):
    err = 1E-6
    log_l_OLD = 0
    diff = np.inf
    N = X.shape[1]


    while diff > err:
        gmm_NEW = []
        S, logdens = logpdf_GMM(X, gmm)
        Post = np.exp(S - logdens)
        Sigma_tied = np.zeros((X.shape[0], X.shape[0]))
        for g in range(len(gmm)):
            gamma = Post[g, :]
            Z_g = np.sum(gamma)
            F_g = np.dot(gamma, X.T)
            S_g = np.dot(X, (vrow(gamma) * X).T)
            w = Z_g / N
            mu = vcol(F_g / Z_g)
            Sigma = S_g / Z_g - np.dot(mu, mu.T)
            Sigma_tied += Z_g*Sigma
            gmm_NEW.append((w,mu))

        gmm = gmm_NEW
        Sigma_tied /= N
        U, s, _ = np.linalg.svd(Sigma_tied)
        s[s < psi] = psi
        Sigma_tied = np.dot(U, vcol(s) * U.T)

        gmm_NEW = []
        for g in range(len(gmm)):
            w, mu = gmm[g]
            gmm_NEW.append((w, mu, Sigma_tied))

        log_l_NEW = logdens.sum() / N
        # print(log_l_NEW)
        diff = np.abs(log_l_NEW - log_l_OLD)

        log_l_OLD = log_l_NEW
        gmm = gmm_NEW

    return gmm

def split_gmm(gmm,alpha=0.1):
    split_gmm = []
    for i in range(len(gmm)):
        U, s, Vh = np.linalg.svd(gmm[i][2])
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        split_gmm.append((gmm[i][0]/2,gmm[i][1]+d,gmm[i][2]))
        split_gmm.append((gmm[i][0]/2,gmm[i][1]-d,gmm[i][2]))
    return split_gmm

def LBG_gmm(X,G,psi=0.01):
    mu, C = mu_and_sigma_ML(X)
    U, s, _ = np.linalg.svd(C)
    s[s < psi] = psi
    C = np.dot(U, vcol(s) * U.T)

    GMM_1 = [(1.0, mu, C)]

    gmm = GMM_1
    while len(gmm) <= G:
        gmm = GMM_EM_diag(X,gmm)
        if len(gmm) == G:
            break
        gmm = split_gmm(gmm)

    return gmm

def Kfold_cross_validation(D, L, K, G, seed=1):
    nSamp = int(D.shape[1]/K)
    residuals = D.shape[1] - nSamp*K
    sub_arr = np.ones((K, 1)) * nSamp
    if residuals != 0:
        sub_arr = np.array([int(x+1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    err = []
    for i in range(K):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i+1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        DTR0 = DTR[:,LTR==0]
        gmm0 = LBG_gmm(DTR0,G)
        DTR1 = DTR[:,LTR==1]
        gmm1 = LBG_gmm(DTR1,G)
        DTR2 = DTR[:,LTR==2]
        gmm2 = LBG_gmm(DTR2,G)

        _, ll0 = logpdf_GMM(DTE,gmm0)
        _, ll1 = logpdf_GMM(DTE, gmm1)
        _, ll2 = logpdf_GMM(DTE, gmm2)

        ll=[]
        ll0 = np.exp(ll0)
        ll1 = np.exp(ll1)
        ll2 = np.exp(ll2)

        ll.append((ll0,ll1,ll2))
        ll = np.reshape(ll, (3,len(ll0)))

        pred = np.argmax(ll,axis=0)
        check = pred == LTE
        acc_i = len(check[check == True]) / len(LTE)
        err.append(1-acc_i)
    return np.mean(err)


if __name__ == '__main__':
    D = np.load('../Data/GMM_data_4D.npy')
    D_1d = np.load('../Data/GMM_data_1D.npy')
    gmm = load_gmm('../Data/GMM_4D_3G_init.json')
    gmm_1D = load_gmm('../Data/GMM_1D_3G_init.json')
    log_densities = np.load('../Data/GMM_4D_3G_init_ll.npy')
    log_densities_1d = np.load('../Data/GMM_1D_3G_init_ll.npy')

    # gmm_tied = GMM_EM_tied(D,gmm)
    # gmm_1D = LBG_gmm(D_1d,4)
    # logdens_our_1D = logpdf_GMM(np.sort(D_1d),gmm_1D)[1]
    # logdens_our = logpdf_GMM(D,gmm)[1]
    # EM_gmm = GMM_EM(D,gmm)
    # gmm_split = LBG_gmm(D,4)
    # print(gmm_split)
    D,L = load_iris()
    print(Kfold_cross_validation(D,L,5,G=4))

    # plt.figure()
    # counts, bins = np.histogram(D_1d,bins=30)
    # plt.hist(bins[:-1],bins,weights=counts,density=True,ec="k")
    # plt.plot(np.sort(D_1d).ravel(),np.exp(logdens_our_1D),'r',linewidth=3)
    # plt.ylim(0,0.4)
    # plt.show()
