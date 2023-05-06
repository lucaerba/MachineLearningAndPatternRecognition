import model
import input
import scipy as sp
import numpy as np

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

    
def main():
    D, L = input.load(input.traininput)
    (DTR, LTR), (DTE, LTE) = input.split_db_2to1(D, L)

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
        print(1-len(check[check==True])/len(LTE))
    
if __name__ == '__main__':
    main()
    
        
    