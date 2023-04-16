import numpy as np
import matplotlib.pyplot as plt
# import os
# print(os.path.abspath("."))

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def logpdf_GAU_ND(x, mu, C):
    logN = []
    M = x.shape[0]
    N = x.shape[1]
    C_inv = np.linalg.inv(C)
    xc = (x - mu)
    x_cent = np.reshape(xc, (M*N, 1), order='F')
    i = 0
    while i in range(N*M):
        xx = x_cent[i:i+M]
        first_term = -.5* M*np.log(2*np.pi)
        second_term = -.5* np.linalg.slogdet(C)[1]
        third_term = -.5* np.dot(vrow(xx),np.ones((M,1))*np.dot(C_inv,vcol(xx)))
        i += M
        logN.append(first_term + second_term + third_term)
    return np.vstack(logN)


def mu_and_sigma_ML(x):
    N = x.shape[1]
    M = x.shape[0]

    mu_ML = []
    sigma_ML = []

    for i in range(M):
        mu_ML.append(np.sum(x[i,:]) / N)

    x_cent = x - np.reshape(mu_ML, (M,1))
    for i in range(M):
        for j in range(M):
            sigma_ML.append(np.dot(x_cent[i,:],x_cent[j,:].T) / N)

    return np.vstack(mu_ML), np.reshape(sigma_ML, (M,M))

def loglikelihood(x, mu_ML, C_ML):
    l = np.sum(logpdf_GAU_ND(x, mu_ML, C_ML))
    return l

plt.figure()
XPlot = np.linspace(-8, 12, 1000)
m = np.ones((1,1)) * 1.0
C = np.ones((1,1)) * 2.0
pdfSol = vcol(np.load('llGAU.npy'))
pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
plt.plot(XPlot.ravel(), np.exp(pdfGau))
# plt.plot(XPlot.ravel(), np.exp(pdfSol))
error = np.abs(pdfSol - pdfGau)
plt.figure()
# plt.plot(XPlot.ravel(), error)
# plt.show()
# print(error.max())

XND = np.load('XND.npy')
mu = np.load('muND.npy')
C = np.load('CND.npy')
pdfSol = vcol(np.load('llND.npy'))
pdfGau = logpdf_GAU_ND(XND, mu, C)
print(np.abs(pdfSol - pdfGau).max())

mu_calc, sigma_calc = mu_and_sigma_ML(XND)
# print(mu_calc, '\n' ,sigma_calc)

ll = loglikelihood(XND, mu_calc, sigma_calc)
# print(ll)

X1D = np.load('X1D.npy')
mu_calc, sigma_calc = mu_and_sigma_ML(X1D)
plt.figure()
plt.hist(X1D.ravel(), bins=50, density=True)
XPlot = np.linspace(-8, 12, 1000)
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu_calc, sigma_calc)))
# plt.show()

ll = loglikelihood(X1D, mu_calc, sigma_calc)
# print(ll)



