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

def dict_freq_tot(dict1, freq1, dict2, freq2, dict3, freq3):
    dictionarytot = {}
    frequenciestot = {}
    eps = 0.001
            
    #-----------------------------------#
    for key in dict1:
        dictionarytot[key] = dict1[key]+eps, 0+eps, 0+eps
        
    for key in dict2:
        if(key in dictionarytot):
            dictionarytot[key] = dict1[key]+eps, dict2[key]+eps, 0+eps
        else:
            dictionarytot[key] = 0+eps, dict2[key]+eps, 0+eps
            
    for key in dict3:
        if(key in dictionarytot):
            if(key in dict1 and key in dict2):
                dictionarytot[key] = dict1[key]+eps, dict2[key]+eps, dict3[key]+eps
            elif(key in dict1):
                dictionarytot[key] = dict1[key]+eps, 0+eps, dict3[key]+eps
            else:
                dictionarytot[key] = 0+eps, dict2[key]+eps, dict3[key]+eps
        else:
            dictionarytot[key] = 0+eps, 0+eps, dict3[key]+eps
    
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
    
    dictionarytot, frequenciestot = dict_freq_tot(dictionaryinf, frequenciesinf, dictionarypur, frequenciespur, dictionarypar, frequenciespar)
    
    #print(frequenciestot)
    
    mat_occurencies =[]
    for k in dictionarytot:
        mat_occurencies.append( vcol(np.array(dictionarytot[k])))
        
    mat_occurencies = np.reshape(mat_occurencies, (np.array(mat_occurencies).shape[0], np.array(mat_occurencies).shape[1]))
    mat_occurencies = np.transpose(mat_occurencies)
    
    mat_frequencies =[]
    for k in frequenciestot:
        mat_frequencies.append( vcol(np.array(frequenciestot[k])))
        
    mat_frequencies = np.reshape(mat_frequencies, (np.array(mat_frequencies).shape[0], np.array(mat_frequencies).shape[1]))
    mat_frequencies = np.transpose(mat_frequencies)
    
    
    #print(mat_occurencies)
    #print(np.array(mat_occurencies).shape)
    #print(mat_frequencies)
    #print(np.array(mat_frequencies).shape)
    
    scores = np.log(mat_frequencies)*mat_occurencies
    
    #print(np.array(list(scores)))
    print(scores)
    
    
    for x in lPar_evaluation:
        p1 = 0
        p2 = 0
        p3 = 0
        print(x)
        for xi in x:
            if(xi in dictionarytot.keys()):
                p1 += scores[0][list(dictionarytot).index(xi)]
                p2 += scores[1][list(dictionarytot).index(xi)]
                p3 += scores[2][list(dictionarytot).index(xi)]
        print(np.argmin([p1,p2,p3]))
        
    
    
    

   
    
    