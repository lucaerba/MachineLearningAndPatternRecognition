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

def vrow(x):
    x = x.reshape((1, x.size))
    return x
def vcol(x):
    x = x.reshape((x.size, 1))
    return x

def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        first_part = 0.5*l*(np.linalg.norm(w))**2
        second_part = 0
        for i in range(DTR.shape[1]):
            second_part += np.logaddexp(0, -(2*LTR[i]-1)*(np.dot(DTR.T[i],w) + b))
        second_part = second_part / DTR.shape[1]
        logreg_obj = first_part + second_part
        return logreg_obj
    return logreg_obj

D, L = load_iris_binary()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

lambdas = [1E-6, 1E-3, 1E-1, 1]
for l in lambdas:
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    # print(DTR)
    (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj,
                                          np.zeros(DTR.shape[0] + 1),
                                          approx_grad = True, maxfun=15000, maxiter=1000)
    # print(x)
    print('lamda: {} -- minimum: {}'.format(l,f))
    # print(d)
    s = np.dot(x[0:-1].T,DTE) + x[-1]
    LP = [1 if s[i]>0 else 0 for i in range(DTE.shape[1])]
    check = LP == LTE
    print(1-len(check[check == True]) / len(LTE))