import numpy
import matplotlib.pyplot as plt
def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    print(v)
    return v

def logpdf_GAU_ND(x, mu, C):
    M = x.size
    cent = x-mu
    r = -M/2*numpy.log10(2*numpy.pi)-0.5*numpy.linalg.slogdet(C)[1]-0.5*cent.transpose()*numpy.linalg.inv(C)*cent
   #print(r)
    return r

plt.figure()
XPlot = numpy.linspace(-8, 12, 1000)
m = numpy.ones((1,1)) * 1.0
C = numpy.ones((1,1)) * 2.0
plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
plt.show()

pdfSol = numpy.load('../Solutions/llGAU.npy')
pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
print(numpy.abs(pdfSol - pdfGau).max())

