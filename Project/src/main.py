import model
import input
import evaluation
import plottering
import testing
import scipy as sp
import numpy as np
import threading

def main(seed=0):
    # D, L = input.load(input.traininput)

    # np.random.seed(seed)
    # idx = np.random.permutation(D.shape[1])
    # D = D[:,idx]
    # L = L[idx]
    # D = D[:,:200]
    # L = L[:200]

    # D, L = input.load_iris_binary()


    #train
    
    # print("logreg...")
    # model.logreg_Znorm_wrapper(D, L)
    # model.logreg_kfold_wrapper(D, L)

    # print("MVG...")
    # model.MVG_kfold_wrapper(D, L)
    # print("NB...")
    # model.NB_kfold_wrapper(D, L)
    # print("TMVG...")
    # model.TMVG_kfold_wrapper(D, L)
    # print("TNB...")
    # model.TNB_kfold_wrapper(D, L)

    # print("GMM...")
    # model.GMM_wrapper(D, L)

    # print("SVM...")
    # model.SVM_wrapper(D, L)

    ## BEST MODELS

    # print('Best models...')
    # model.actual_DCF_and_scores(D,L)
    # model.calibration_wrapper(D, L)
    # model.fusion_wrapper(D, L)
    # model.DET_plot(D,L)

    #evaluation

    DTR_fin, LTR_fin = input.load(input.traininput)
    DTE_fin, LTE_fin = input.load(input.testinput)
    np.random.seed(seed)
    idx_train = np.random.permutation(DTR_fin.shape[1])
    idx_test = np.random.permutation(DTE_fin.shape[1])

    DTR_fin = DTR_fin[:, idx_train]
    LTR_fin = LTR_fin[idx_train]
    DTE_fin = DTE_fin[:, idx_test]
    LTE_fin = LTE_fin[idx_test]

    # DTR_fin = DTR_fin[:,:200]
    # LTR_fin = LTR_fin[:200]
    # DTE_fin = DTE_fin[:,:300]
    # LTE_fin = LTE_fin[:300]

    print('testing...')
    # testing.evaluation_test(DTR_fin, DTE_fin, LTR_fin, LTE_fin)
    # testing.DET_plot_evaluation(DTR_fin, DTE_fin, LTR_fin, LTE_fin)
    # testing.Bayes_plot_evaluation(DTR_fin, DTE_fin, LTR_fin, LTE_fin)
    # testing.optimal_wrap_evaluation(DTR_fin, DTE_fin, LTR_fin, LTE_fin, model='MVG')
    testing.fusion_wrapper(DTR_fin, LTR_fin, DTE_fin, LTE_fin)
       
if __name__ == '__main__':
    main()
    
        
    