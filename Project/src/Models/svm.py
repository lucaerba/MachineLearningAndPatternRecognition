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
        
        print("minDCF: {}".format(evaluation.minDCF(scores, self.L, 0.1, 1, 1)))

