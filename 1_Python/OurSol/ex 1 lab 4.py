import numpy as np
import matplotlib.pyplot as plt

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def logpdf_GAU_ND(x, mu, C):
    logN = []
    M = x.shape[0]
    C_inv = np.linalg.inv(C)
    x_cent = vcol(x - mu)

    for i in range(x.shape[1]):
        first_term = -.5* M*np.log(2*np.pi)
        second_term = -.5* np.linalg.slogdet(C)[1]
        third_term = -.5* x_cent[i]*C_inv*x_cent[i]

        logN.append(first_term + second_term + third_term)
    return np.vstack(logN)

plt.figure()
XPlot = np.linspace(-8, 12, 1000)
m = np.ones((1,1)) * 1.0
C = np.ones((1,1)) * 2.0
pdfSol = np.load('llGAU.npy')
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
plt.plot(XPlot.ravel(), np.exp(pdfSol))
plt.show()

pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
print(pdfGau)
print(pdfSol)
print(np.abs(pdfSol - pdfGau).max())

XND = np.load('XND.npy')
mu = np.load('muND.npy')
C = np.load('CND.npy')
pdfSol = np.load('llND.npy')
pdfGau = logpdf_GAU_ND(XND, mu, C)
# print(pdfGau)
# print(pdfSol)
# print(np.abs(pdfSol - pdfGau).max())
