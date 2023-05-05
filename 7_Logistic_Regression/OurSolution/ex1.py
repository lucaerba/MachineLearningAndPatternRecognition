import scipy as sp
import numpy as np

#f(y,z) = (y + 3)^2 + sin(y) + (z + 1)^2 

def f(y, z):
    return np.power(y + 3, 2) + np.sin(y) + np.power(z + 1, 2)
def objective(x):
    y = x[0]
    z = x[1]
    return f(y, z)

(x, f, d) = sp.optimize.fmin_l_bfgs_b(objective, np.array([0,0]), approx_grad = True)

print(x)
print(f)
print(d)
