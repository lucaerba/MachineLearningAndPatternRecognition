import scipy as sp
import numpy as np    

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

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

def logreg_wrapper(DTR, LTR, DTE, LTE):
    lam = [10**-6, 10**-3, 10**-2, 10**-1, 1, 100, 1000, 10000]
    for l in lam:
        (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = True, args=(DTR, LTR, l))
        
        print("lam " + str(l) + " min:" + str(f))
        
        #print(x,d)
        S = np.dot(x[0:-1].T, DTE) + x[-1]
        #print(S)    
        S_sc = [1 if S[i]>0 else 0 for i in range(len(DTE.T))]
        #print(S_sc)
        
        check = S_sc==LTE
        check2 = [True if (not check[i] and S_sc[i] == 0) else False for i in range(len(DTE.T))]
        check2 = [val for val in check2 if val == True]
        check3 = [True if (not check[i] and S_sc[i] == 1) else False for i in range(len(DTE.T))]
        check3 = [val for val in check3 if val == True]
        
        print(1-len(check[check==True])/len(LTE))
        print("Test len:"+str(len(LTE))+" FN:"+str(len(check2))+ " FP:"+ str(len(check3))) 
        
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
    for i in range(K):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i+1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        Sjoint = func(DTR, LTR, DTE)
        pred, acc_i = predicted_labels_and_accuracy(Sjoint, LTE)
        err.append(1-acc_i)
    return np.min(err), pred

K = 5
def MVG_kfold_wrapper(D, L):
    print("MVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_MVG)))
    
def NB_kfold_wrapper(D, L):
    print("NB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_NaiveBayes)))
    
def TMVG_kfold_wrapper(D, L):
    print("TMVG err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedMVG)))
    
def TNB_kfold_wrapper(D, L):
    print("TNB err: "+str(Kfold_cross_validation(D, L, K, func=score_matrix_TiedNaiveBayes)))
    