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
############################## GMM ##############################

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

def LBG_gmm(X,G,func,psi=0.01):
    mu, C = mu_and_sigma_ML(X)
    U, s, _ = np.linalg.svd(C)
    s[s < psi] = psi
    C = np.dot(U, vcol(s) * U.T)

    GMM_1 = [(1.0, mu, C)]

    gmm = GMM_1
    while len(gmm) <= G:
        gmm = func(X,gmm)
        if len(gmm) == G:
            break
        gmm = split_gmm(gmm)

    return gmm

def Kfold_cross_validation_GMM(D, L, K, G_Target, G_nonTarget, pi=0.1, C_fn=1, C_fp=1, func=GMM_EM, seed=1):
    nSamp = int(D.shape[1]/K)
    residuals = D.shape[1] - nSamp*K
    sub_arr = np.ones((K, 1)) * nSamp
    if residuals != 0:
        sub_arr = np.array([int(x+1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    err = []
    minDCF = []
    for i in range(K):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i+1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        DTR0 = DTR[:,LTR==0]
        gmm0 = LBG_gmm(DTR0,G_nonTarget,func)
        DTR1 = DTR[:,LTR==1]
        gmm1 = LBG_gmm(DTR1,G_Target,func)

        _, ll0 = logpdf_GMM(DTE, gmm0)
        _, ll1 = logpdf_GMM(DTE, gmm1)

        ll=[]
        ll0 = np.exp(ll0)
        ll1 = np.exp(ll1)

        ll.append((ll0,ll1))
        ll = np.reshape(ll, (2,len(ll0)))

        pred = np.argmax(ll,axis=0)
        check = pred == LTE
        acc_i = len(check[check == True]) / len(LTE)

        llr = ll1-ll0
        minDCF.append(evaluation.minDCF(llr,LTE,pi,C_fn,C_fp))

        err.append(1-acc_i)

    return np.mean(err), np.mean(minDCF)

def mu_and_sigma_ML(x):
    N = x.shape[1]

    mu_ML = np.mean(x,axis=1)
    x_cent = x - mu_ML[:, np.newaxis]

    sigma_ML = 1/N * np.dot(x_cent,x_cent.T)

    return mu_ML, sigma_ML

def logpdf_GAU_ND(X, mu, C):
    XC = X - vcol(mu)
    M = X.shape[0]
    const = - 0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC*np.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v
