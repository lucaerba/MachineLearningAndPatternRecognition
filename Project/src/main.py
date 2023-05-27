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

    model.logreg_kfold_wrapper(D,L)
    # model.MVG_kfold_wrapper(D, L)
    # model.NB_kfold_wrapper(D, L)
    # model.TMVG_kfold_wrapper(D, L)
    # model.TNB_kfold_wrapper(D, L)
    
if __name__ == '__main__':
    main()
    
        
    