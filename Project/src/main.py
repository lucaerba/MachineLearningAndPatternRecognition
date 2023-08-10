import model
import input
import evaluation
import plottering
import scipy as sp
import numpy as np
import threading

P_target = 0.1
C_fn = 1
C_fa = 1

def DCF(FN, FP):
    return P_target * C_fn * FN + (1 - P_target) * C_fa * FP

#min DCF, PCA, plot
def main(seed=1,K=5):
    D, L = input.load(input.traininput)

    #plot
    plottering.plot_simple()
    plottering.plot_LDA(D,L)
    plottering.plot_correlations(D)
    plottering.plot_Scatter(D,L)

    #train
    
    model.logreg_kfold_wrapper(D, L)
    model.MVG_kfold_wrapper(D, L)
    model.NB_kfold_wrapper(D, L)
    model.TMVG_kfold_wrapper(D, L)
    model.TNB_kfold_wrapper(D, L)
    model.SVM_wrapper(D, L)
    model.GMM_wrapper(D, L)
     
    #calibration

    #evaluation
       
if __name__ == '__main__':
    main()
    
        
    