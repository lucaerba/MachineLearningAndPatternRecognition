import numpy as np
import scipy as sp
import sklearn.datasets

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def load_iris_binary(K=1):
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    D = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    return D, L

def matrix_H(D,label):
    z = np.where(label == 0,1,0)
    G = np.dot(D.T,D)
    H = np.dot(z.T,z)*G
    return H

def L_wrap(D,H):
    def L_and_gradL(alpha):
        alpha = np.reshape(alpha, (D.shape[1],1))
        L_d = 0.5*alpha.T*np.dot(H,alpha.T) - alpha.T*np.ones((1,D.shape[1]))
        grad_L = np.dot(H,alpha.T) - 1
        grad_L = np.reshape(grad_L, (D.shape[1],))
        return L_d, grad_L
    return L_and_gradL

def linear_SVM(D,H):
    C = [.1,1,10]
    L_d, grad_L = L_wrap(D,H)
    for c in C:
        (x,f,d) = sp.optimize.fmin_l_bfgs_b(L_d,
                                            np.zeros((1,D.shape[1])),
                                            fprime=grad_L,
                                            bounds=[(0, c) for _ in range(D.shape[1])],
                                            maxfun=15000, maxiter=1000)
        print(x)
        print(f)
        print(d)

if __name__ == '__main__':
    D,L = load_iris_binary()
    H = matrix_H(D,L)
    linear_SVM(D,H)