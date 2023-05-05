import sklearn.datasets
import scipy as sp
import numpy as np

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
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
w_f, b_f = [], 0
def logreg_obj(v, DTR, LTR, l):
    w, b = v[0:-1], v[-1]
    w_f, b_f = w, b
    return J(w, b, DTR, LTR, l)
#-----------------------------------------#
D, L = load_iris_binary()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

lam = [10**-6, 10**-3, 10**-1, 1]
for l in lam:
    (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = True, args=(DTR, LTR, l))
    print("lam " + str(l) + " min:" + str(f))
    #print(x,d)
    S = np.dot(x[0:-1].T, DTE) + x[-1]
    #print(S)    
    S_sc = [1 if S[i]>0 else 0 for i in range(len(DTE.T))]
    #print(S_sc)
    check = S_sc==LTE
    print(1-len(check[check==True])/len(LTE))
    