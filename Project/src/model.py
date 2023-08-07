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

################################### LOG-REG ##########################################

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
        minDCF = []
        for i in range(K):
            idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i + 1]))]
            idxTrain = [x for x in idx if x not in idxTest]
            DTR = D[:, idxTrain]
            DTE = D[:, idxTest]
            LTR = L[idxTrain]
            LTE = L[idxTest]
            (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj,
                                                  np.zeros(DTR.shape[0] + 1),
                                                  approx_grad = True,
                                                  args=(DTR, LTR, l))
        
        # print("lam " + str(l) + " min:" + str(f))
        
        #print(x,d)
            S = np.dot(x[0:-1].T, DTE) + x[-1]
        #print(S)    
            S_sc.append([1 if S[i]>0 else 0 for i in range(len(DTE.T))])
        #print(S_sc)
        
            check = S_sc[i]==LTE

            err.append(1 - len(check[check == True]) / len(LTE))

            minDCF.append(evaluation.minDCF(S, LTE, 0.1, 1, 1))
        # print(l)
        print('Lambda = {} -- minDCF: {}'.format(l,np.mean(minDCF)))
    # return np.mean(err), S_sc[np.argmin(err)], np.mean(minDCF), l


############################## MVG and NAIVE + TIED ###################################
        
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
    minDCF = []
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

        scores = - Sjoint[0,:] + Sjoint[1,:]

        minDCF.append(evaluation.minDCF(scores, LTE, 0.1, 1, 1))

    return np.min(err),pred[np.argmin(err)],np.mean(minDCF)

############################## SVM ##############################

class Kernel:

    def __init__(self, d=2, c=1, g=2, const=1):
       self.d = d
       self.c = c
       self.g = g
       self.eps = 0
       self.const = const
    
    def polynomial(self, x1, x2):
       return (np.dot(x1.T, x2) + self.const) ** self.d + self.eps
   
    def rbf_kernel(self, x1, x2):
        kernel = np.zeros((x1.shape[1], x2.shape[1]))
        for i in range(x1.shape[1]):
            for j in range(x2.shape[1]):
                kernel[i, j] = np.exp(-self.g * (np.linalg.norm(x1[:, i] - x2[:, j]) ** 2)) + self.eps
        return kernel

class SVM:
    def __init__(self, D, L, c, fun):
        self.D = D
        self.L = L
        self.c = c
        self.fun = fun
        
    def matrix_H_kernel(self):
        z = vcol(np.where(self.L == 0, 1, -1))
        # H = np.empty((len(self.L), len(self.L)))
        # D = self.D.T
        # for i in range(D.shape[0]):
        #     for j in range(D.shape[0]):
        #         H[i][j] = z[i] * z[j] * self.fun(D[i], D[j])
        #
        D = self.D
        H = vcol(z) * vrow(z) * self.fun(D,D)

        self.H = H
        
    def L_and_gradL(self, alpha):
        L_d = 0.5 * np.dot(alpha.T, np.dot(self.H, alpha)) - np.dot(alpha, np.ones(np.shape(alpha)[0]))
        return L_d

    def grad(self, alpha):
        grad_L = np.dot(self.H, alpha) - np.ones(self.D.shape[1])
        return np.reshape(grad_L, (self.D.shape[1],))

    def exec(self):
        self.matrix_H_kernel()
        z = np.where(self.L == 0, 1, -1)
        # print(z)

        (x, f, d) = sp.optimize.fmin_l_bfgs_b(self.L_and_gradL,
                                            np.zeros((self.D.shape[1], 1)),
                                            approx_grad=False,
                                            fprime=self.grad,
                                            bounds=[(0, self.c) for _ in range(self.D.shape[1])],
                                            maxfun=15000, maxiter=100000,
                                            factr=1.0)

        scores = np.dot(x*z, self.fun(self.D, self.D))
        # D = self.D.T
        # scores2 = np.empty(len(self.L))
        # for i in range(D.shape[0]):
        #     scores2[i] = 0
        #     for j in range(D.shape[0]):
        #         scores2[i] = scores2[i] + x[j] * z[j] * self.fun(D[j], D[i])

        check = np.where(scores > 0, 1, -1) == z
        
        print(" Dual loss =  " + str(-f))
        print(" " + str(float(1 - len(check[check == True]) / len(z))) + " err")
        
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
    minDCF = []
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

        llr = ll1-ll0
        minDCF.append(evaluation.minDCF(llr,LTE,0.1,1,1))

        err.append(1-acc_i)

    return np.mean(err), np.mean(minDCF)

K = 5
def MVG_kfold_wrapper(D, L):
    # print("MVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_MVG)))
    for m in range(2,11):
        DP = PCA(D,m)
        print('PCA = {}'.format(m))
        print('MVG -- minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_MVG)[2]))
    
def NB_kfold_wrapper(D, L):
    # print("NB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_NaiveBayes)))
    for m in range(2,11):
        DP = PCA(D,m)
        print('PCA = {}'.format(m))
        print('NB -- minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_NaiveBayes)[2]))
    
def TMVG_kfold_wrapper(D, L):
    # print("TMVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedMVG)))
    for m in range(2,11):
        DP = PCA(D,m)
        print('PCA = {}'.format(m))
        print('TMVG -- minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_TiedMVG)[2]))
    
def TNB_kfold_wrapper(D, L):
    # print("TNB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedNaiveBayes)))
    for m in range(2,11):
        DP = PCA(D,m)
        print('PCA = {}'.format(m))
        print('TNB -- minDCF: {}'.format(Kfold_cross_validation(DP, L, K, func=score_matrix_TiedNaiveBayes)[2]))

def logreg_kfold_wrapper(D, L):
    print('-- Logistic Regression -- (no PCA)')
    # print("LogReg err: "+str(logreg_wrapper(D,L)[0]))
    logreg_wrapper(D,L)
    for m in range(2,11):
        print('-- Logistic Regression -- (PCA = {})'.format(m))
        DP = PCA(D, m)
        logreg_wrapper(DP, L)
    
def SVM_wrapper(D, L):
    cs = [10**-5, 2*(10**-5), 5*(10**-5)]
    D = np.append(D, np.ones((1,D.shape[1])),axis=0)
    #polynomial
    for c in cs:
        for mul in [1, 10, 100, 1000]:
            c_val = c * mul
            pol_kern = Kernel(d=2)
            svm = SVM(D, L, c_val, pol_kern.polynomial)
            print("c= "+str(c_val)+" poly("+str(2)+")")
            svm.exec()

            pol_kern = Kernel(d=3)
            svm = SVM(D, L, c_val, pol_kern.polynomial)
            print("c= "+str(c_val)+" poly("+str(3)+")")
            svm.exec()

    print("------------------")
    #rbf
    for c in cs:
        for mul in [1, 10, 100, 1000]:
            c_val = c * mul
            
            rbf_kern = Kernel(g=2)
            svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
            print("c= "+str(c_val)+" rbf("+str(2)+")")
            svm.exec()
            
            rbf_kern = Kernel(g=3)
            svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
            print("c= "+str(c_val)+" rbf("+str(3)+")")
            svm.exec()
            
            rbf_kern = Kernel(g=4)
            svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
            print("c= "+str(c_val)+" rbf("+str(4)+")")
            svm.exec()
            
            rbf_kern = Kernel(g=5)
            svm = SVM(D, L, c_val, rbf_kern.rbf_kernel)
            print("c= "+str(c_val)+" rbf("+str(5)+")")
            svm.exec()
    """ #linear
    for c in cs:
        for mul in [1, 10, 100, 1000]:
            c = c * mul
            H = matrix_H_kernel(D, L, linear)
            SVM(D, L, H, c, linear) """
    
    
    
def GMM_wrapper(D, L):
    G = [1,2,4,8,16]
    functions = [GMM_EM, GMM_EM_diag, GMM_EM_tied]

    for g in G:
        print('------ g = {} ------'.format(g))
        for f in functions:
            err, minDCF = Kfold_cross_validation_GMM(D, L, K, g, f)
            # print("error: {}, using function: {} ".format(err,f.__name__))
            # print(pred[np.argmin(err)])

            print('function = {}, minDCF: {}'.format(f.__name__,minDCF))



