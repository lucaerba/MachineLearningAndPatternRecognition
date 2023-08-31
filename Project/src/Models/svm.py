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

    def linear(x1, x2):
        return np.dot(x1.T, x2)
    
    def polynomial(self, x1, x2):
        return (np.dot(x1.T, x2) + self.const) ** self.d + self.eps
   
    def rbf_kernel(self, x1, x2):
        kernel = np.zeros((x1.shape[1], x2.shape[1]))
        for i in range(x1.shape[1]):
            for j in range(x2.shape[1]):
                kernel[i, j] = np.exp(-self.g * (np.linalg.norm(x1[:, i] - x2[:, j]) ** 2)) + self.eps
        return kernel

class SVM:
    def __init__(self, D, L, c, K, fun):
        self.D = D
        self.L = L
        self.c = c
        self.K = K
        self.fun = fun
        
    def matrix_H_kernel(self, DTR, LTR):
        z = np.where(LTR == 1, 1, -1)
        H = vcol(z) * vrow(z) * self.fun(DTR,DTR)
        self.H = H
        
    def L_d(self, alpha):
        L_d = 0.5 * np.dot(vrow(alpha), np.dot(self.H, vcol(alpha))).ravel() - alpha.sum()
        return L_d

    def grad(self, alpha):
        grad_L = np.dot(self.H, vcol(alpha)).ravel() - np.ones(alpha.size)
        return grad_L

    def scores(self, DTR, LTR, DTE):
        z = np.where(LTR == 1, 1, -1)
        self.matrix_H_kernel(DTR, LTR)

        x, f, d = sp.optimize.fmin_l_bfgs_b(self.L_d,
                                            np.zeros(DTR.shape[1]),
                                            approx_grad=False,
                                            fprime=self.grad,
                                            bounds=[(0, self.c)] * DTR.shape[1],
                                            maxfun=100000, maxiter=100000,
                                            factr=1.0)
        omega_star = np.dot(DTR, vcol(x) * vcol(z))

        scores = np.dot(omega_star.T, DTE)
        scores = scores.ravel()
        return scores

    def exec(self, K_fold=5, seed=1, pi=0.1, C_fn=1, C_fp=1):
        nSamp = int(self.D.shape[1] / K_fold)
        residuals = self.D.shape[1] - nSamp * K_fold
        sub_arr = np.ones((K_fold, 1)) * nSamp

        if residuals != 0:
            sub_arr = np.array([int(x + 1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
        np.random.seed(seed)
        idx = np.random.permutation(self.D.shape[1])

        # err = []
        minDCF = []
        for i in range(K_fold):
            idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i + 1]))]
            idxTrain = [x for x in idx if x not in idxTest]
            DTR = self.DTR = self.D[:, idxTrain]
            DTE = self.DTE = self.D[:, idxTest]
            LTR = self.LTR = self.L[idxTrain]
            LTE = self.LTE = self.L[idxTest]

            DTR = np.vstack((DTR, self.K * np.ones(DTR.shape[1])))
            DTE = np.vstack((DTE, self.K * np.ones(DTE.shape[1])))

            scores = SVM.scores(self, DTR, LTR, DTE)

            minDCF.append(evaluation.minDCF(scores, LTE, pi, C_fn, C_fp))

            # predicted_labels = np.where(scores > 0, 1, 0)
            # check = predicted_labels == LTE
            # acc_i = len(check[check == True]) / len(LTE)
            # err.append(1 - acc_i)

        return np.mean(minDCF)
        
        # print("minDCF: {} -- WP = ({}, {}, {})".format(np.mean(minDCF) , pi, C_fn, C_fp))
        # print('error = {}'.format(np.mean(err)))

