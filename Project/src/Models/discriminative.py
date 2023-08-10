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
################################### LOG-REG ##########################################

K = 5

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
def logreg_wrapper(D,L,l,seed=0):
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
    return np.mean(err), S_sc[np.argmin(err)], np.mean(minDCF), l

def vec(x):
    x = vcol(x)
    return np.vstack(np.dot(x,x.T))

def QUAD_log_reg(D,L,l,seed=0):
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

        DTR_vec = np.apply_along_axis(vec, 0, DTR)[0,:,:]
        PHI_DTR = np.vstack([DTR_vec, DTR])

        DTE_vec = np.apply_along_axis(vec, 0, DTE)[0,:,:]
        PHI_DTE = np.vstack([DTE_vec, DTE])

        (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj,
                                              np.zeros(PHI_DTR.shape[0] + 1),
                                              approx_grad=True,
                                              args=(PHI_DTR, LTR, l))

        # print("lam " + str(l) + " min:" + str(f))

        # print(x,d)
        S = np.dot(x[0:-1].T, PHI_DTE) + x[-1]
        # print(S)
        S_sc.append(np.where(S > 0, 1, 0))
        # print(S_sc)

        check = S_sc[i] == LTE

        err.append(1 - len(check[check == True]) / len(LTE))

        minDCF.append(evaluation.minDCF(S, LTE, 0.1, 1, 1))
    # print(l)
    return np.mean(err), S_sc[np.argmin(err)], np.mean(minDCF), l