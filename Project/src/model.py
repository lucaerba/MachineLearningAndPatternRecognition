import scipy as sp
import numpy as np    
import threading

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def PCA(D,m): # m = leading eigenvectors
    N = len(D)
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = N ** -1 * np.dot(DC, DC.T)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    return DP

def LDA(D,L,m,N_classes = 2):
    N = len(D[0])
    S_W = 0
    S_B = 0
    mu = np.array(np.mean(D, 1))

    for i in range(N_classes):
        D_class = D[:, L == i]
        n_c = len(D_class[0])
        mu_class = np.array(np.mean(D_class, 1))
        DC_class = D_class - vcol(mu_class)
        C_class = n_c ** -1 * np.dot(DC_class, DC_class.T)

        S_W += C_class * n_c
        S_B += n_c * np.dot(vcol(mu_class) - vcol(mu), (vcol(mu_class) - vcol(mu)).T)
    S_W = S_W / N
    S_B = S_B / N

    s, U = sp.linalg.eigh(S_B, S_W)
    W = U[:, ::-1][:, 0:m]

    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]

    DP = np.dot(W.T, D)

    return DP

def J(w, b, DTR, LTR, l):
    #first = l/2*np.square(np.linalg.norm(w))
    #second = np.sum(np.logaddexp(0, -(2*LTR-1)*(np.transpose(w)*DTR+b)))*(1/len(DTR))
     # Compute the regularizer term np.sum(np.power(w, 2))
    reg_term = (l/2) * np.square(np.linalg.norm(w))

    # Compute the logistic loss term
    NEW_DTR = np.transpose(DTR)
    n = len(NEW_DTR)
    loss_term = 0
    for i in range(n):
        loss_term += np.logaddexp(0,-(2 * LTR[i] - 1) * (np.dot(NEW_DTR[i], np.transpose(w)) + b))
    loss_term = loss_term*1/n
    # Compute the full objective function
    objective = reg_term + loss_term
    return objective

def logreg_obj(v, DTR, LTR, l):
    w, b = v[0:-1], v[-1]

    return J(w, b, DTR, LTR, l)
#-----------------------------------------#

def logreg_wrapper(D,L,seed=0):
    lam = [10**-6, 10**-3, 10**-2, 10**-1, 1, 100, 1000, 10000]
    for l in lam:
        nSamp = int(D.shape[1] / K)
        residuals = D.shape[1] - nSamp * K
        sub_arr = np.ones((K, 1)) * nSamp
        if residuals != 0:
            sub_arr = np.array([int(x + 1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
        np.random.seed(seed)
        idx = np.random.permutation(D.shape[1])
        err = []
        S_sc = []
        err_l_min = []
        pred_l_min = []
        for i in range(K):
            idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i + 1]))]
            idxTrain = [x for x in idx if x not in idxTest]
            DTR = D[:, idxTrain]
            DTE = D[:, idxTest]
            LTR = L[idxTrain]
            LTE = L[idxTest]
            (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = True, args=(DTR, LTR, l))
        
        # print("lam " + str(l) + " min:" + str(f))
        
        #print(x,d)
            S = np.dot(x[0:-1].T, DTE) + x[-1]
        #print(S)    
            S_sc.append([1 if S[i]>0 else 0 for i in range(len(DTE.T))])
        #print(S_sc)
        
            check = S_sc[i]==LTE

            err.append(1 - len(check[check == True]) / len(LTE))
        # print(l)
        err_l_min.append(np.min(err))
        pred_l_min.append(S_sc[np.argmin(err)])
        
    return (err_l_min),(pred_l_min)

        
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
        
    return np.min(err),pred[np.argmin(err)]

############################## SVM ##############################


#TODO eps or c after every kern fun

d = 2
gamma = 2
eps = 0
        
def set_g(myg):
    gamma = myg
        
def polynomial_kernel(x1, x2):
    return (np.dot(x1.T, x2) + 1) ** d + eps

def rbf_kernel(x1, x2):
    pairwise_dist = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * pairwise_dist) + eps


def linear(x1, x2):
    return np.dot(x1.T, x2)
        
def SVM(D, label, c, fun=linear):
    H = np.empty((len(label), len(label)))

    def matrix_H_kernel(D, label, fun=linear):
        z = vcol(np.where(label == 0, 1, -1))
        H = np.empty((len(label), len(label)))
        D = D.T
        for i in range(D.shape[0]):
            for j in range(D.shape[0]):
                H[i][j] = z[i] * z[j] * fun(D[i], D[j])

        return H

    def L_and_gradL(alpha):
        L_d = 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.dot(alpha, np.ones(np.shape(alpha)[0]))
        return L_d

    def grad(alpha):
        grad_L = np.dot(H, alpha) - np.ones(D.shape[1])
        return np.reshape(grad_L, (D.shape[1],))

        
    H = matrix_H_kernel(D, label, fun=linear)
    z = np.where(label == 0, 1, -1)
    # print(z)

    (x, f, d) = sp.optimize.fmin_l_bfgs_b(L_and_gradL,
                                        np.zeros((D.shape[1], 1)),
                                        approx_grad=False,
                                        fprime=grad,
                                        bounds=[(0, c) for _ in range(D.shape[1])],
                                        maxfun=15000, maxiter=100000,
                                        factr=1.0)

    # scores = np.dot(x*z, fun(D, D))
    D = D.T
    scores2 = np.empty(len(label))
    for i in range(D.shape[0]):
        scores2[i] = 0
        for j in range(D.shape[0]):
            scores2[i] = scores2[i] + x[j] * z[j] * fun(D[j], D[i])

    check = np.where(scores2 > 0, 1, -1) == z
    
    print(" Dual loss =  " + str(-f))
    print(str(float(1 - len(check[check == True]) / len(z))) + " % err")

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

def Kfold_cross_validation_GMM(D, L, K, G, func=GMM_EM, seed=1):
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
        gmm0 = LBG_gmm(DTR0,G,func)
        DTR1 = DTR[:,LTR==1]
        gmm1 = LBG_gmm(DTR1,G,func)

        _, ll0 = logpdf_GMM(DTE,gmm0)
        _, ll1 = logpdf_GMM(DTE, gmm1)

        ll=[]
        ll0 = np.exp(ll0)
        ll1 = np.exp(ll1)

        ll.append((ll0,ll1))
        ll = np.reshape(ll, (2,len(ll0)))

        pred = np.argmax(ll,axis=0)
        check = pred == LTE
        acc_i = len(check[check == True]) / len(LTE)
        err.append(1-acc_i)
    return np.min(err)

K = 5
def MVG_kfold_wrapper(D, L):
    print("MVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_MVG)))
    
def NB_kfold_wrapper(D, L):
    print("NB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_NaiveBayes)))
    
def TMVG_kfold_wrapper(D, L):
    print("TMVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedMVG)))
    
def TNB_kfold_wrapper(D, L):
    print("TNB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedNaiveBayes)))

def logreg_kfold_wrapper(D, L):
    print("LogReg err: "+str(logreg_wrapper(D,L)))
    
def GMM_wrapper(D, L):
    G = [1,2,4,8,16,32]
    functions = [GMM_EM, GMM_EM_diag, GMM_EM_tied]
    for g in G:
        print('------ g = {} ------'.format(g))
        for f in functions:
            err = Kfold_cross_validation_GMM(D, L, 5, g, f)
            print("error: {}, using function: {} ".format(err,f.__name__))

