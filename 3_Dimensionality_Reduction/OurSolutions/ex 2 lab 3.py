import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

input_file = sys.argv[1]

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def load(file):
    attributes = []
    fam_list = []
    families = {
        'Iris-setosa' : 0,
        'Iris-versicolor' : 1,
        'Iris-virginica' : 2
    }

    with open(file, 'r') as file:
        for line in file:
            vector = line.split(',')
            flower_fam = vector.pop(-1).strip()
            fam_ind = families[flower_fam]
            vector = np.array([float(i) for i in vector]).reshape(len(vector),1)
            attributes.append(vector)
            fam_list.append(fam_ind)

    return np.hstack(attributes), np.array((fam_list))

def LDA(D,m,N_classes = 2):
    N = len(D[0])
    S_W = 0
    S_B = 0
    mu = np.array(np.mean(D, 1))

    for i in range(N_classes):
        D_class = D[:, L == i]
        n_c = len(D_class[0])
        mu_class = np.array(np.mean(D_class, 1))
        DC_class = D_class - vcol(mu_class)
        C_class = n_c ** -1 * np.dot(DC_class, DC_class.T)

        S_W += C_class * n_c
        S_B += n_c * np.dot(vcol(mu_class) - vcol(mu), (vcol(mu_class) - vcol(mu)).T)
    S_W = S_W / N
    S_B = S_B / N

    s, U = sp.linalg.eigh(S_B, S_W)
    W = U[:, ::-1][:, 0:m]

    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]

    DP = np.dot(W.T, D)

    return DP

m = 2 # number of leading eigenvectors
D, L = load(input_file)
N = len(D[0])
S_W = 0
S_B = 0
mu = np.array(np.mean(D, 1))

for i in range(3):
    D_class = D[:, L == i]
    n_c = len(D_class[0])
    mu_class = np.array(np.mean(D_class, 1))
    DC_class = D_class - vcol(mu_class)
    C_class = n_c**-1 * np.dot(DC_class,DC_class.T)

    S_W += C_class*n_c
    S_B += n_c*np.dot(vcol(mu_class)-vcol(mu),(vcol(mu_class)-vcol(mu)).T)
S_W = S_W/N
S_B = S_B/N

## SOLUTION WITH GENERALIZED EIGENVALUE PROBLEM ##

s, U = sp.linalg.eigh(S_B, S_W)
W = U[:, ::-1][:, 0:m]

UW, _, _ = np.linalg.svd(W)
U = UW[:, 0:m] # orthogonal base for the subspace spaced by W

## SOLUTION BY JOINT DIAGONALIZATION OF SB AND SW ##

# U, s, _ = np.linalg.svd(S_W)
# P1 = np.dot(U * vrow(1.0/(s**0.5)), U.T)


DP = np.dot(W.T, D) # we apply the projection

DP0 = [DP[0][0:50], DP[1][0:50]]
DP1 = [DP[0][50:100], DP[1][50:100]]
DP2 = [DP[0][100:151], DP[1][100:151]]

plt.figure()
plt.scatter(DP0[0], DP0[1])
plt.scatter(DP1[0], DP1[1])
plt.scatter(DP2[0], DP2[1])
plt.show()


