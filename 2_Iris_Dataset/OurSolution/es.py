import sys
import numpy as np
import matplotlib.pyplot as plt

def load():
    retmat = []
    retvet = []
    with open("iris.csv") as f:
        
        for l in f:
            line = []
            v1,v2,v3,v4,name = l.split(",")
            name = name[0:-1]
            if (name == "Iris-setosa"):
                name = 0
            elif (name == "Iris-versicolor"):
                name = 1
            elif (name == "Iris-virginica"):
                name = 2

            line = [float(v1),float(v2),float(v3),float(v4)]
            retvet.append(int(name))
            retmat.append(line)

    return np.array(retmat), np.array(retvet)


dataset, labels = load()

M0 = labels == 0
M1 = labels == 1
M2 = labels == 2

D0 = dataset[M0]    
D1 = dataset[M1]    
D2 = dataset[M2]    

b = 30
for i in range(4):        
    plt.figure()
    plt.hist(D0[:,i], bins=b, color = '#800000', alpha=0.5, density = True)   
    plt.hist(D1[:,i], bins=b, color = '#008000', alpha=0.5, density = True)   
    plt.hist(D2[:,i], bins=b, color = '#000080', alpha=0.5, density = True)
    
plt.show()  
mu = 0
for i in range(dataset.shape[1]):
    mu = mu + dataset[:, i:i+1]
    mu = mu / float(dataset.shape[1])

print(mu)
mu = dataset.mean(1)
print(mu)
DC = dataset - dataset.mean(1).reshape((dataset.shape[0], 1))
print(DC)