import sys
import numpy as np
import matplotlib.pyplot as plt

input_file = sys.argv[1]
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

D, L = load(input_file)

D0 = D[:, L == 0]
D1 = D[:, L == 1]
D2 = D[:, L == 2]
bins = 10

features = {
    0: 'sepal length [cm]',
    1: 'sepal width [cm]',
    2: 'petal length [cm]',
    3: 'petal width [cm]'
}

for i in range(4):

    plt.figure()
    plt.xlabel(features[i])
    plt.hist(D0[i,:], density=True, bins=bins, alpha=0.4, label='Iris-setosa')
    plt.hist(D1[i, :], density=True, bins=bins, alpha=0.4, label='Iris-versicolor')
    plt.hist(D2[i, :], density=True, bins=bins, alpha=0.4, label='Iris-virginica')
    plt.legend()


for i in range(4):
    ii_list = []
    for a in range(4):
        if a != i:
            ii_list.append(a)
    for ii in ii_list:
        plt.figure()
        plt.xlabel(features[i])
        plt.ylabel(features[ii])
        plt.scatter(D0[i, :], D0[ii, :], label='Iris-setosa')
        plt.scatter(D1[i, :], D1[ii, :], label='Iris-versicolor')
        plt.scatter(D2[i, :], D2[ii, :], label='Iris-virginica')
    plt.legend()
plt.show()