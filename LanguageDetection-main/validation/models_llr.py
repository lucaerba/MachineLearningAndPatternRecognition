#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All models in this file returns llr vectors in order to use it to compute minDCF for evaluation

Created on Wed May 31 12:10:21 2023
@author: guido
"""
import sys
sys.path.append("../")
import numpy as np
import scipy.optimize
import Functions as f

Nc = 2

#MVG

def MVG_log(DTR, LTR, DTE, LTE):
    DTR_c = [DTR[:, LTR == i] for i in range(Nc)]
    m_c = []
    s_c = []
    for d in DTR_c:
        m_c.append(d.mean(1))
        s_c.append(np.cov(d, bias=True))

    S = np.zeros((Nc, DTE.shape[1]))

    for i in range(Nc):
        for j, sample in enumerate(DTE.T):
            sample = f.vcol(sample)
            S[i, j] = f.logpdf_GAU_ND(sample, f.vcol(m_c[i]), s_c[i])

    #logSJoint = np.log(1/Nc) + S
    #logSSum = scipy.special.logsumexp(logSJoint, axis=0)
    #logSPost = logSJoint - logSSum
 
    #llr = np.array([S[1,i] - S[0,i] for i in range(S.shape[1])])   
    llr = S[1,:] - S[0,:]
    
    return llr

def NaiveBayesGaussianClassifier(DTR, LTR, DTE, LTE):
    DTR_c = [DTR[:, LTR == i] for i in range(Nc)]
    m_c = []
    s_c = []
    for d in DTR_c:
        m_c.append(d.mean(1))
        s_c.append(np.cov(d, bias=True)*np.identity(d.shape[0]))

    S = np.zeros((Nc, DTE.shape[1]))

    for i in range(Nc):
        for j, sample in enumerate(DTE.T):
            sample = f.vcol(sample)
            S[i, j] = f.logpdf_GAU_ND(sample, f.vcol(m_c[i]), s_c[i])
    
    # SJoint = np.log(1/Nc) + S
    # SSum = scipy.special.logsumexp(SJoint, axis=0)
    # SPost = SJoint - SSum
    
    llr = S[1,:] - S[0,:]
    return llr
    

def TiedCovarianceGaussianClassifier(DTR, LTR, DTE, LTE):
    DTR_c = [DTR[:, LTR == i] for i in range(Nc)]
    m_c = []
    s_c = []
    for d in DTR_c:
        m_c.append(d.mean(1))
        s_c.append(np.cov(d, bias=True))

    SStar = 0
    for i in range(Nc):
        SStar += DTR_c[i].shape[1]*s_c[i]
    SStar /= DTR.shape[1]

    S = np.zeros((Nc, DTE.shape[1]))

    for i in range(Nc):
        for j, sample in enumerate(DTE.T):
            sample = f.vcol(sample)
            S[i, j] = f.logpdf_GAU_ND(sample, f.vcol(m_c[i]), SStar)

    # SJoint = np.log(1/Nc) + S
    # SSum = scipy.special.logsumexp(SJoint, axis=0)
    # SPost = SJoint - SSum

    llr = S[1,:] - S[0,:]
    return llr

#LOGREG

def logreg_obj_wrap(DTR, LTR, l, pi=None):
    """Wrappper function for binary logistic regression.
       Returns the function that implement the binary log-reg that we have to minimize
       If pi is none the logreg will be not weigthed, if pi is passed the logreg will be prior-weigthed."""
    if(pi == None):
        def logreg_obj(v):
            w, b = v[0:-1], v[-1]
            n = DTR.shape[1]  # number of samples in training set
            s = 0
            for i in range(n):
                xi = DTR[:, i]
                zi = (2*LTR[i]) - 1
                s += np.logaddexp(0, -zi*(np.dot(w.T, xi) + b))
            return l/2 * np.linalg.norm(w) ** 2 + s/n
    else:
        def logreg_obj(v):
            w, b = v[0:-1], v[-1]
            # number of samples in training set for each class
            nt = DTR[:, LTR == 1].shape[1]
            nf = DTR.shape[1] - nt
            n = nt+nf
            st = 0
            sf = 0
            for i in range(n):
                xi = DTR[:, i]
                zi = (2*LTR[i]) - 1
                if(zi == 1):
                    st += np.logaddexp(0, -zi*(np.dot(w.T, xi) + b))
                elif(zi == -1):
                    sf += np.logaddexp(0, -zi*(np.dot(w.T, xi) + b))
            reg = l/2 * np.linalg.norm(w) ** 2
            return reg + (pi/nt) * st + ((1-pi)/nf) * sf

    return logreg_obj

def logistic_regression(DTR, LTR, l, DTE, LTE, pi=None, cal=True):
    # instantiate objective function and...
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, pi)
    # ...minimize it:
    minimizer = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True)

    # get w and b parameters computed with minimization
    w, b = minimizer[0][0:-1], minimizer[0][-1]

    # compute a score for each test sample Xt where score = (w.T * Xt) + b
    score = np.dot(w.T,DTE) + b
    
    if(cal and pi != None):
    #calibrate scores in order to obtain llr in output
        c = np.log(pi / (1 - pi))
        llr = score - c
        
    return llr

# SVM

def SVM_linear(DTR, LTR, DTE, C, K):
    # build the extended training data matrix D_ext
    D_ext = np.vstack((DTR, K * np.ones(DTR.shape[1])))

    # compute the zi for each train sample, z = 1 if class is 1, -1 if class is 0
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # compute H_hat
    H = np.dot(D_ext.T,D_ext)
    H = f.vcol(Z) * f.vrow(Z) * H
    
    def JDual(alpha):
        Ha = np.dot(H,f.vcol(alpha))
        aHa = np.dot(f.vrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = np.dot(f.vrow(w), D_ext)
        loss = np.maximum(np.zeros(S.shape), 1-Z*S).sum()
        return 0.5 * np.linalg.norm(w)**2 + C * loss
    
    alphaStar , _x, _y = scipy.optimize.fmin_l_bfgs_b(
    LDual,
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(D_ext, f.vcol(alphaStar) * f.vcol(Z))
    
    # build the extended evaluation data matrix DTEEXT
    DTEEXT = np.vstack((DTE,np.array([K for i in range(DTE.shape[1])])))
    # make scores
    Scores = np.dot(wStar.T,DTEEXT)
    
    return Scores.ravel()

def SVM_Poly(DTR, LTR, DTE, C, K, d, c):
    
    # compute the zi for each train sample, z = 1 if class is 1, -1 if class is 0
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # Compute H_hat directly on training data, no expansion needed. 
    # and compute H using kernel function instead of dot product
    H = ((np.dot(DTR.T,DTR) + c) ** d) + K * K
    H = f.vcol(Z) * f.vrow(Z) * H
    
    def JDual(alpha):
        Ha = np.dot(H,f.vcol(alpha))
        aHa = np.dot(f.vrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    alphaStar , _x, _y = scipy.optimize.fmin_l_bfgs_b(
    LDual,
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(DTR, f.vcol(alphaStar) * f.vcol(Z))
    
    kernel = ((np.dot(DTR.T,DTE) + c) ** d) + K * K #kernel based on product between TRAINING and TEST samples
    scores = np.sum(np.dot(alphaStar * f.vrow(Z), kernel), axis=0)
    return scores.ravel() # ravel to get a 1D array (N,) instead of a 2D (1,N) 

def SVM_RBF(DTR, LTR, DTE, C, K, gamma):
    
    # compute the zi for each train sample, z = 1 if class is 1, -1 if class is 0
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    # Compute H_hat directly on training data, no expansion needed. 
    # and compute H using kernel function instead of dot product
    H = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            H[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = f.vcol(Z) * f.vrow(Z) * H
    
    def JDual(alpha):
        Ha = np.dot(H,f.vcol(alpha))
        aHa = np.dot(f.vrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    

    alphaStar , _x, _y = scipy.optimize.fmin_l_bfgs_b(
    LDual,
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(DTR, f.vcol(alphaStar) * f.vcol(Z))
    
    # compute kernel on train and TEST set
    kernel = np.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kernel[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K
    
    scores = np.sum(np.dot(alphaStar * f.vrow(Z), kernel), axis=0)
    return scores.ravel()
    
# GMM
def mean_and_covarianceMatrix(D):
    """Compute mean and covariance matrix for a given Dataset D where each column is a sample"""
    N = D.shape[1]  # total number of samples in the dataset (each sample is a column)
    # compute mean by column of the dataset for each dimension, notice that D.mean(1) is a row vector
    mu = f.vcol(D.mean(1))
    DC = D - mu  # center data
    C = np.dot(DC, DC.T)/N  # compute the covariance matrix
    return mu, C

def logpdf_GAU_ND(X,mu,C) :
    
    res = -0.5*X.shape[0]*np.log(2*np.pi)
    res += -0.5*np.linalg.slogdet(C)[1]
    res += -0.5*((X-mu)*np.dot(np.linalg.inv(C), (X-mu))).sum(0) 
    return res

def logpdf_GMM(X, gmm):
    
    SJ = np.zeros((len(gmm),X.shape[1]))
    
    for g, (w, mu, C) in enumerate(gmm):
        SJ[g,:] = logpdf_GAU_ND(X, mu, C) + np.log(w)

    SM = scipy.special.logsumexp(SJ, axis=0)
    
    return SJ, SM #Note: use SM to compute then llr -> SM class 1 - SM class 0, we use logpdf!

def GMM_EM(X, gmm):
    '''
    EM algorithm for GMM full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (f.vrow(gamma)*X).sum(1)
            S = np.dot(X, (f.vrow(gamma)*X).T)
            w = Z/N
            mu = f.vcol(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = np.dot(U, f.vcol(s)*U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        #print(llNew)
    #print(llNew-llOld)
    return gmm

def GMM_EM_diag(X, gmm):
    '''
    EM algorithm for GMM diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (f.vrow(gamma)*X).sum(1)
            S = np.dot(X, (f.vrow(gamma)*X).T)
            w = Z/N
            mu = f.vcol(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            #diag
            Sigma = Sigma * np.eye(Sigma.shape[0])
            U, s, _ = np.linalg.svd(Sigma)
            s[s<psi] = psi
            sigma = np.dot(U, f.vcol(s)*U.T)
            gmmNew.append((w, mu, sigma))
        gmm = gmmNew
        #print(llNew)
    #print(llNew-llOld)
    return gmm

def GMM_EM_tied(X, gmm):
    '''
    EM algorithm for GMM tied covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    #sigma_list = []
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        
        sigmaTied = np.zeros((X.shape[0],X.shape[0]))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (f.vrow(gamma)*X).sum(1)
            S = np.dot(X, (f.vrow(gamma)*X).T)
            w = Z/N
            mu = f.vcol(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            sigmaTied += Z*Sigma
            gmmNew.append((w, mu))
        #get tied covariance
        gmm = gmmNew
        sigmaTied = sigmaTied/N
        U,s,_ = np.linalg.svd(sigmaTied)
        s[s<psi]=psi 
        sigmaTied = np.dot(U, f.vcol(s)*U.T)
        
        gmmNew=[]
        for g in range(G):
            (w,mu)=gmm[g]
            gmmNew.append((w,mu,sigmaTied))
        gmm=gmmNew
        
        #print(llNew)
    #print(llNew-llOld)
    return gmm

def GMM_LBG(X, doub, version):
    assert version == 'full' or version == 'diagonal' or version == 'tied', "GMM version not correct"
    init_mu, init_sigma = mean_and_covarianceMatrix(X)
    gmm = [(1.0, init_mu, init_sigma)]
    
    for i in range(doub):
        doubled_gmm = []
        
        for component in gmm: 
            w = component[0]
            mu = component[1]
            sigma = component[2]
            U, s, Vh = np.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * 0.1 # 0.1 is alpha
            component1 = (w/2, mu+d, sigma)
            component2 = (w/2, mu-d, sigma)
            doubled_gmm.append(component1)
            doubled_gmm.append(component2)
            if version == "full":
                gmm = GMM_EM(X, doubled_gmm)
            elif version == "diagonal":
                gmm = GMM_EM_diag(X, doubled_gmm)
            elif version == "tied":
                gmm = GMM_EM_tied(X, doubled_gmm)
            
    return gmm