import scipy as sp
import numpy as np
import threading
import evaluation
from prettytable import PrettyTable
from Models.discriminative import *
from Models.gaussians import *
from Models.gmm import *
from Models.svm import *
import sys
import time
import os

mvg_output_file = '../Out_VALIDATION/mvg_output.txt'

def PCA(D,m, cal=False): # m = leading eigenvectors
    N = len(D)
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = N ** -1 * np.dot(DC, DC.T)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    if cal == False:
        return DP
    else:
        return DP, P

