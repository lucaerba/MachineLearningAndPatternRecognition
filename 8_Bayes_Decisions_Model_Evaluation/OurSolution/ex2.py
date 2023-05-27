import sys
import numpy as np
sys.path.append('/Users/gabrieleiob/Desktop/python/Machine Learning/MachineLearningAndPatternRecognition/6_Generative_Models_II/OurSolutions')
import ex_1_lab_6 as dante

if __name__ == '__main__':
    lInf, lPur, lPar = dante.load_data()
    lInf_train, lInf_evaluation = dante.split_data(lInf, 4)
    lPur_train, lPur_evaluation = dante.split_data(lPur, 4)
    lPar_train, lPar_evaluation = dante.split_data(lPar, 4)

    dictionaryinf, frequenciesinf = dante.dictionaryAndfreq(lInf_train)
    dictionarypur, frequenciespur = dante.dictionaryAndfreq(lPur_train)
    dictionarypar, frequenciespar = dante.dictionaryAndfreq(lPar_train)

    dictionarytot, frequenciestot = dante.dict_freq_tot(dictionaryinf, dictionarypur, dictionarypar)

    mat_occurencies, mat_frequencies = dante.matrices(dictionarytot, frequenciestot)

    eval = [lInf_evaluation, lPur_evaluation, lPar_evaluation]

    accuracies, pred = dante.evaluation(dictionarytot, eval, mat_frequencies)

    indices = {}
    for i in range(3):
        indices[i] = [i]*len(pred[i])
    confusion_matrix = np.zeros((3, 3))
    for ii in range(len(pred)):
        for word in range(len(pred[ii])):
            confusion_matrix[pred[ii][word]][indices[ii][word]] += 1
    print(confusion_matrix)