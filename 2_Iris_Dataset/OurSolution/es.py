import sys
import numpy as np

def load():
    finput = "iris.csv" 
    retmat = []
    retvet = []
    with open(finput) as f:
        line = []
        v1,v2,v3,v4,name = f.readline().split(",")
        
        if (name == "Iris-setosa"):
            name = 0
        elif (name == "Iris-versicolor"):
            name = 1
        elif (name == "Iris-virginica"):
            name = 2

        line = [v1,v2,v3,v4]
        retvet.append(name)
        retmat.append(line)

    
    return np.array(retmat), np.array(retvet)


dataset, labels = load()
print(dataset)
print(labels)


