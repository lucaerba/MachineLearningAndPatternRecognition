import sys
import numpy as np
import matplotlib.pyplot as plt

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def PCA(D,m):
    N = len(D)
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = N ** -1 * np.dot(DC, DC.T)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    return DP

input_file = sys.argv[1]
D = []

with open(input_file, 'r') as f:
    for line in f:
        data_vector = line.split(',')
        data_vector.pop(-1).strip()
        data_vector = np.array([float(i) for i in data_vector]).reshape(len(data_vector),1)
        D.append(data_vector)

N = len(D)
m = 2 # number of leading eigenvectors

D = np.hstack(D) # data matrix
mu = np.array(np.mean(D, 1)) # mean values
DC = D - vcol(mu)

C = N**-1 * np.dot(DC,DC.T) # covariance matrix

s, U = np.linalg.eigh(C) # eigenvalues and eigenvectors in descending order

P = U[:, ::-1][:, 0:m] # retrieve m leading eigenvectors

DP = np.dot(P.T, D) # we apply the projection


DP0 = [DP[0][0:50], DP[1][0:50]]
DP1 = [DP[0][50:100], DP[1][50:100]]
DP2 = [DP[0][100:151], DP[1][100:151]]

plt.figure()
plt.scatter(DP0[0], DP0[1])
plt.scatter(DP1[0], DP1[1])
plt.scatter(DP2[0], DP2[1])
# plt.show()
