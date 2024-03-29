import scipy as sp
import numpy as np
import sklearn.datasets
testinput = "../Data/Test.txt"
traininput = "../Data/Train.txt"

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
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

def load(file):
    attributes = []
    fam_list = []

    with open(file, 'r') as file:
        for line in file:
            vector = line.split(',')
            fam_ind = int(vector.pop(-1).strip())
            vector = np.array([float(i) for i in vector]).reshape(len(vector),1)
            attributes.append(vector)
            fam_list.append(fam_ind)

    return np.hstack(attributes), np.array((fam_list))