import model
import input
import evaluation
import scipy as sp
import numpy as np
import threading

P_target = 0.1
C_fn = 1
C_fa = 1

def DCF(FN, FP):
    return P_target * C_fn * FN + (1 - P_target) * C_fa * FP

def main(seed=1,K=5):
    D, L = input.load(input.traininput)

    # Create threads for each function
    threads = []

    #threads.append(threading.Thread(target=model.logreg_kfold_wrapper, args=(D, L)))
    #threads.append(threading.Thread(target=model.MVG_kfold_wrapper, args=(D, L)))
    #threads.append(threading.Thread(target=model.NB_kfold_wrapper, args=(D, L)))
    #threads.append(threading.Thread(target=model.TMVG_kfold_wrapper, args=(D, L)))
    #threads.append(threading.Thread(target=model.TNB_kfold_wrapper, args=(D, L)))
    threads.append(threading.Thread(target=model.SVM_wrapper, args=(D, L)))
    
    # Start the threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
     
    DTE, LTE = input.load(input.testinput)
    
    nSamp = int(D.shape[1]/K)
    residuals = D.shape[1] - nSamp*K
    sub_arr = np.ones((K, 1)) * nSamp

    if residuals != 0:
        sub_arr = np.array([int(x+1) for x in sub_arr[:residuals]] + [int(x) for x in sub_arr[residuals:]])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    for i in range(K):
        idxTest = idx[int(np.sum(sub_arr[:i])):int(np.sum(sub_arr[:i+1]))]
        idxTrain = [x for x in idx if x not in idxTest]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]

        S = model.score_matrix_MVG(DTR, LTR, DTE)
        pred_l, acc = model.predicted_labels_and_accuracy(S, LTE)

        print("---------"+str(i)+"-------------")
        print(" ConfMatr: "+str(evaluation.optimal_Bayes_decision(0.1, 1, 1, pred_l, LTE)))
        print(" DCF: "+str(evaluation.Bayes_risk(0.1, 1, 1, pred_l, LTE)))
        print(" err: "+str((1-acc)*100)+" % ")
        
if __name__ == '__main__':
    main()
    
        
    