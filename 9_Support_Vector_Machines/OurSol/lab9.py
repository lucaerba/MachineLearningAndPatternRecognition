import numpy as np
import scipy as sp
import sklearn.datasets

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def matrix_H(D,label):
    z = vcol(np.where(label == 0,1,-1))
    G = np.dot(D.T,D)
    H = np.dot(z,z.T)*G
    return H

def Jw(w,d,z,c):
    first = 0.5*np.linalg.norm(w)**2
    
    second = np.sum(np.maximum(0, 1-z*np.dot(d.T, w)))
    
    return first + c*second

def linear_SVM(D,label,c,H):
    def L_and_gradL(alpha):
        L_d = 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.dot(alpha,np.ones(np.shape(alpha)[0]))
        return L_d

    def grad(alpha):
        grad_L = np.dot(H, alpha) - np.ones(D.shape[1])
        return np.reshape(grad_L, (D.shape[1],))

    z = np.where(label == 0, 1, -1)
    #print(z)

    (x,f,d) = sp.optimize.fmin_l_bfgs_b(L_and_gradL,
                                        np.zeros((D.shape[1],1)),
                                        approx_grad=False,
                                        fprime=grad,
                                        bounds=[(0, c) for _ in range(D.shape[1])],
                                        maxfun=15000, maxiter=100000,
                                        factr=1.0)
    
    
    w = np.dot(D,x*z)
    
    scores = np.dot(w.T,D)
    
    check = np.where(scores>0,1,-1)==z
    
    print(str(1-len(check[check==True])/len(z)))
    print("c = "+str(c)+" - ("+str(Jw(w, D, z, c))+", "+str(-f)+") - duality gap: "+ str(Jw(w, D, z, c)+f))

c_my = 0
d_my = 2
g_my = 1
eps = 0

def polynomial_kernel(x1, x2, c=c_my, d=d_my):
    c=c_my
    d=d_my
    return (np.dot(x1.T, x2) + c) ** d + eps

def rbf_kernel(x1, x2, gamma=g_my):
    gamma=g_my
    pairwise_dist = np.linalg.norm(x1-x2)**2
    return np.exp(-gamma * pairwise_dist) + eps

def linear(x1, x2):
    return np.dot(x1.T, x2)

def matrix_H_kernel(D, label, fun=linear):
    z = vcol(np.where(label == 0,1,-1))
    H = np.empty((100,100))
    D = D.T
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            H[i][j]= z[i]*z[j]*fun(D[i], D[j])

    return H

def SVM(D,label,H, c, fun=linear):
    def L_and_gradL(alpha):
        L_d = 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.dot(alpha,np.ones(np.shape(alpha)[0]))
        return L_d

    def grad(alpha):
        grad_L = np.dot(H, alpha) - np.ones(D.shape[1])
        return np.reshape(grad_L, (D.shape[1],))

    z = np.where(label == 0, 1, -1)
    #print(z)
    
    (x,f,d) = sp.optimize.fmin_l_bfgs_b(L_and_gradL,
                                        np.zeros((D.shape[1],1)),
                                        approx_grad=False,
                                        fprime=grad,
                                        bounds=[(0, c) for _ in range(D.shape[1])],
                                        maxfun=15000, maxiter=100000,
                                        factr=1.0)
    
    
    #scores = np.dot(x*z, fun(D, D))
    D = D.T
    scores2 = np.empty(100)
    for i in range(D.shape[0]):
        scores2[i] = 0
        for j in range(D.shape[0]):
            scores2[i] = scores2[i] + x[j]*z[j]*fun(D[j], D[i])
        
    check = np.where(scores2>0,1,-1)==z
    
    print(str(float(1-len(check[check==True])/len(z)))+" % err")
    print("c = "+str(c)+", K = "+str(K)+" ,dual loss =  "+str(-f))

if __name__ == '__main__':
    D,L = load_iris_binary()
    #print(D)
    
    K=1
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, polynomial_kernel)
    SVM(Dk, L, H, 1, polynomial_kernel)
    
    
    K=10
    eps=3
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, polynomial_kernel)
    SVM(Dk, L, H, 1, polynomial_kernel)
    
    K=1
    eps=1
    c_my=1
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, polynomial_kernel)
    SVM(Dk, L, H, 1, polynomial_kernel)
    
    K=10
    eps=3
    c_my=1
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, polynomial_kernel)
    SVM(Dk, L, H, 1, polynomial_kernel)

    K=1
    eps=1
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, rbf_kernel)
    SVM(Dk, L, H, 1, rbf_kernel)
    
    g_my = 10
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, rbf_kernel)
    SVM(Dk, L, H, 1, rbf_kernel)
    
    K=10
    eps=3
    g_my = 1
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, rbf_kernel)
    SVM(Dk, L, H, 1, rbf_kernel)
    
    g_my = 10
    Dk = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
    H = matrix_H_kernel(Dk,L, rbf_kernel)
    SVM(Dk, L, H, 1, rbf_kernel) 
    """    
    for K in [1, 10]:
        D = np.append(D, K*np.ones((1,D.shape[1])),axis=0)
        for c in [0.1, 1, 10]:
            H = matrix_H(D,L)
            linear_SVM(D, L, c, H)  Praveen 404-2332948-2437146
"""
 
