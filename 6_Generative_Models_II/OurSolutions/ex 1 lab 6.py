import sys
import numpy as np

def load_data():

    lInf = []

    f=open('../Data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f=open('../Data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f=open('../Data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()
    
    return lInf, lPur, lPar

def split_data(l, n):

    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
            
    return lTrain, lTest

def dictionaryAndfreq(trainset):
    dictionary = {}

    for line in trainset:
        for word in line.split(" "):
            if (word in dictionary.keys()):
                dictionary[word] += 1
            else:
                dictionary[word] = 1
        
    N = sum(dictionary.values())
    
    frequencies = {}
    for key in dictionary:
        frequencies[key] = dictionary[key]/N
        
    return dictionary, frequencies

def dict_freq_tot(dict1, dict2, dict3):
    dictionarytot = {}
    frequenciestot = {}
    eps = 0.001

    all_keys = list(set(list(dict1) + list(dict2) + list(dict3)))
    for key in all_keys:
        try:
            val1 = dict1[key]
        except KeyError:
            val1 = 0

        try:
            val2 = dict2[key]
        except KeyError:
            val2 = 0

        try:
            val3 = dict3[key]
        except KeyError:
            val3 = 0

        dictionarytot[key] = val1 + eps, val2 + eps, val3 + eps

    N1 = sum(dict1.values())
    N2 = sum(dict2.values())
    N3 = sum(dict3.values())
    
    for k in dictionarytot:
        frequenciestot[k] = dictionarytot[k][0]/N1, dictionarytot[k][1]/N2, dictionarytot[k][2]/N3
         
    return dictionarytot, frequenciestot

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

if __name__ == '__main__':

    # Load the tercets and split the lists in training and test lists
    
    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)
    
    dictionaryinf, frequenciesinf = dictionaryAndfreq(lInf_train)
    dictionarypur, frequenciespur = dictionaryAndfreq(lPur_train)
    dictionarypar, frequenciespar = dictionaryAndfreq(lPar_train)
    
    dictionarytot, frequenciestot = dict_freq_tot(dictionaryinf, dictionarypur, dictionarypar)
    
    # print(len(dictionarytot))
    
    mat_occurencies = []
    for k in dictionarytot:
        mat_occurencies.append( vcol(np.array(dictionarytot[k])))
        
    mat_occurencies = np.reshape(mat_occurencies, (np.array(mat_occurencies).shape[0], np.array(mat_occurencies).shape[1]))
    mat_occurencies = np.transpose(mat_occurencies)
    
    mat_frequencies = []
    for k in frequenciestot:
        mat_frequencies.append( vcol(np.array(frequenciestot[k])))
        
    mat_frequencies = np.reshape(mat_frequencies, (np.array(mat_frequencies).shape[0], np.array(mat_frequencies).shape[1]))
    mat_frequencies = np.transpose(mat_frequencies)
    
    
    # print(mat_occurencies)
    #print(np.array(mat_occurencies).shape)
    # print(mat_frequencies)
    #print(np.array(mat_frequencies).shape)
    
    #print(np.array(list(scores)))
    # print(scores)

    eval = [lInf_evaluation, lPur_evaluation, lPar_evaluation]
    cantiche = [0, 1, 2]
    for ii in cantiche:
        predicted_indices = []
        for x in eval[ii]:
            L1 = 0
            L2 = 0
            L3 = 0
            # print(x)
            for xi in x.split(" "):
                if xi in dictionarytot.keys():
                    ind = list(dictionarytot).index(xi)
                    L1 += np.log(mat_frequencies[0,ind])
                    L2 += np.log(mat_frequencies[1,ind])
                    L3 += np.log(mat_frequencies[2,ind])
            predicted_indices.append(np.argmax([L1,L2,L3]))

        check = [i for i in predicted_indices if i == cantiche[ii]]
        print(len(check)/len(predicted_indices))

   
    
    