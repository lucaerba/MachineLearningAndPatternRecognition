import sklearn.datasets
import scipy as sp
import numpy as np

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
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
        first_part = 0.5*l*(w*w).sum()
        second_part = np.mean(np.logaddexp(0, -(2*LTR-1)*(np.dot(DTR.T,w) + b)))
        logreg = first_part + second_part
        return logreg
    return logreg_obj

def logreg_obj_wrap_multiclass(DTR, LTR, l):
    def logreg_obj(v):
        v = np.reshape(v, (DTR.shape[0] + 1, N_classes))
        b = np.reshape(v[-1,:], (N_classes,1))
        W = v[0:-1,:]
        S = np.dot(W.T, DTR) + b
        first = 0.5 * l * np.linalg.norm(W)**2
        logsumexp = np.apply_along_axis(func1d=np.logaddexp.reduce, axis=0, arr=S)
        Y_log = S - logsumexp
        T = np.array([LTR == i for i in range(N_classes)]).astype(int)
        second = np.mean((T * Y_log).sum(0))
        return first - second
    return logreg_obj

def binary_logistic(DTR, LTR, DTE, LTE):
    lambdas = [1E-6, 1E-3, 1E-1, 1]
    for l in lambdas:
        logreg_obj = logreg_obj_wrap(DTR, LTR, l)
        (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj,
                                              np.zeros(DTR.shape[0] + 1),
                                              approx_grad=True, maxfun=15000, maxiter=1000)
        print('lamda: {} -- minimum: {:.3e}'.format(l, f))
        s = np.dot(x[0:-1].T, DTE) + x[-1]
        LP = np.where(s > 0, 1, 0)
        check = LP == LTE
        print('err: {:.3f}'.format(1 - len(check[check == True]) / len(LTE)))

def multiclass_logistic(DTR, LTR, DTE, LTE, N_classes):
    lambdas = [1E-6, 1E-3, 1E-1, 1]
    for l in lambdas:
        logreg_obj = logreg_obj_wrap_multiclass(DTR, LTR, l)
        (x, f, d) = sp.optimize.fmin_l_bfgs_b(logreg_obj,
                                              np.zeros((DTR.shape[0] + 1, N_classes)),
                                              approx_grad=True, maxfun=15000, maxiter=1000)
        print('lamda: {} -- minimum: {:.3e}'.format(l, f))
        x = np.reshape(x, (DTR.shape[0] + 1, N_classes))
        b = x[-1, :]
        w = x[0:-1, :]
        s = np.dot(w.T, DTE) + np.array(b).reshape(3, 1)
        LP = np.argmax(s, axis=0)
        check = LP == LTE
        print('err: {:.3f}'.format(1 - len(check[check == True]) / len(LTE)))


D, L = load_iris_binary()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
binary_logistic(DTR, LTR, DTE, LTE)
print('---------------------')
print('---------------------')
D, L = load_iris()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
N_classes = 3
multiclass_logistic(DTR, LTR, DTE, LTE, N_classes)


