import model
import input
import scipy as sp
import numpy as np
P_target = 0.1
C_fn = 1
C_fa = 1
def DCF(FN, FP):
    return P_target * C_fn * FN + (1 - P_target) * C_fa * FP

def main():
    D, L = input.load(input.traininput)
    (DTR, LTR), (DTE, LTE) = input.split_db_2to1(D, L)

    lam = [10**-6, 10**-3, 10**-2, 10**-1, 1, 100, 1000, 10000]
    for l in lam:
        (x, f, d) = sp.optimize.fmin_l_bfgs_b(model.logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = True, args=(DTR, LTR, l))
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
    
if __name__ == '__main__':
    main()
    
        
    