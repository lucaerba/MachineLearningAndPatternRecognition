#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:05:56 2023

This file contains minDCF and actual DCF functions

@author: guido
"""

import numpy as np
import matplotlib.pyplot as plt

def comp_confmat(labels, predicted):
    """Compute confusion matrix.
       Inputs are:
       - labels: labels of samples, each position is a sample, the content is the relative label
       - predicted: a np vector containing predictions of samples, structured as labels"""
    # extract the different classes
    classes = np.unique(labels)
    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))
    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):
           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((labels == classes[i]) & (predicted == classes[j]))
    return confmat.T.astype(int)

def compute_binaryOptimalBayesDecisions(pi,Cfn,Cfp,classifier_CP, th = None):
    """Compute bayes optimal decision based on passed parameters.
       Class posterior probabilities (test scores) obtained from a binary classifier are passed with classifier_CP.
    """
    if th == None:
        th = -np.log((pi*Cfn)/((1-pi)*Cfp))
    OptDecisions = np.array([classifier_CP > th])
    return np.int32(OptDecisions)

def compute_emp_Bayes_binary(M, pi, Cfn, Cfp):
    """Compute the empirical bayes risk, assuming thath M is a confusion matrix of a binary case"""
    FNR = M[0,1]/(M[0,1] + M[1,1])
    FPR = M[1,0]/(M[0,0] + M[1,0])
    return pi*Cfn*FNR + (1-pi)*Cfp*FPR # bayes risk's formula for binary case 

def compute_normalized_emp_Bayes_binary(M, pi, Cfn, Cfp):
    """Compute the normalized bayes risk, assuming thath M is a confusion matrix of a binary case"""
    empBayes = compute_emp_Bayes_binary(M,pi,Cfn,Cfp) 
    B_dummy = np.array([pi*Cfn, (1-pi)*Cfp]).min()
    return empBayes / B_dummy

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    """Compute the actual DCF, which is basically the normalized bayes risk"""
    #compute opt bayes decisions
    Pred = compute_binaryOptimalBayesDecisions(pi, Cfn, Cfp, scores, th=th)
    #compute confusion matrix
    CM = comp_confmat(labels, Pred)
    #compute DCF and return it
    return compute_normalized_emp_Bayes_binary(CM, pi, Cfn, Cfp)

def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    """Compute the minDCF"""
    t = np.array(scores)
    t.sort()
    np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return np.array(dcfList).min()

#TODO testare e vedere se funziona
def DET_plot(scores, labels):
    #compute the set of threasholds from test scores, order them in increasing order and keep unique values
    thresholds = np.array(scores)
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    
    #initialize verctors of FPR and TPR as numpy array, we will use these to plot ROC curve
    FPR = np.zeros(thresholds.size)
    FNR = np.zeros(thresholds.size)
        
    
    for idx,t in enumerate(thresholds):
        # compute opt bayes decisions
        predictions = np.int32(scores>t)

        # compute confusion matrix
        m = comp_confmat(labels,predictions)

        # *** extract FNR and FPR for each considered t ***
        FPR[idx] = m[1,0]/(m[0,0] + m[1,0])
        #TPR[idx] = 1 - m[0,1]/(m[0,1] + m[1,1])
        FNR[idx] = m[0,1]/(m[0,1]+m[1,1]) 
        
    #plot ROC curve
    plt.plot(FPR,FNR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    return plt
    
def bayes_error_plot(scores,labels, title):
    # compute the p-tilde values
    effPriorLogOdds = np.linspace(-3, 3, 21)
    
    # initialize DCF and minDCF vectors that will be plotted
    DCF = np.zeros(effPriorLogOdds.size)
    minDCF = np.zeros(effPriorLogOdds.size)
    
    # set Cfn and Cfp to 1
    Cfn = 1
    Cfp = 1
    
    #compute DCF and minDCF for each value of p-tilde considered (21 in total)
    for idx, p_tilde in enumerate(effPriorLogOdds):
        
        #compute prior π
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        
        # compute DCF
        DCF[idx] = compute_act_DCF(scores, labels, pi_tilde, Cfn, Cfp)
        
        # compute minDCF
        minDCF[idx] = compute_min_DCF(scores, labels, pi_tilde, Cfn, Cfp)
            
    plt.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.xlabel(r'$log\frac{\pi}{1-\pi}$')
    plt.ylabel("DCF")
    plt.legend()
    plt.grid()
    plt.title(title)
    
    # the following will be done in scripts that use this function:
    # plt.ylim([0, 0.4])
    # plt.xlim([-3, 3])
    # plt.savefig("filename", dpi = 200)
    # plt.show()
   
    return plt
    
def bayes_error_plot_comparison(scores, labels, model, plot = None, color = None):
    """Plot a bayes error plot fro the first time if plot=None, else means that a plot containing another bayes error plot from another model has been passed
        so the new one will be added 
    """
    # compute the p-tilde values
    effPriorLogOdds = np.linspace(-3, 3, 50)
    
    # initialize DCF and minDCF vectors that will be plotted
    DCF = np.zeros(effPriorLogOdds.size)
    minDCF = np.zeros(effPriorLogOdds.size)
    
    # set Cfn and Cfp to 1
    Cfn = 1
    Cfp = 1
    
    #compute DCF and minDCF for each value of p-tilde considered (21 in total)
    for idx, p_tilde in enumerate(effPriorLogOdds):
        
        #compute prior π
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        
        # compute DCF
        DCF[idx] = compute_act_DCF(scores, labels, pi_tilde, Cfn, Cfp)
        
        # compute minDCF
        minDCF[idx] = compute_min_DCF(scores, labels, pi_tilde, Cfn, Cfp)
    
    if plot == None:
        # create a plot here
        plt.plot(effPriorLogOdds, DCF, label=f'{model} DCF', color='r')
        plt.plot(effPriorLogOdds, minDCF, '--', label=f'{model} minDCF', color='r')
        plt.xlabel(r'$log\frac{\pi}{1-\pi}$')
        plt.ylabel("DCF")
        return plt
    else:
        # a plot has been passed, add current model DCF and minDCF to it
        plot.plot(effPriorLogOdds, DCF, label=f'{model} DCF', color=color)
        plot.plot(effPriorLogOdds, minDCF, '--', label=f'{model} minDCF', color=color)
        plot.xlabel(r'$log\frac{\pi}{1-\pi}$')
        return plot  
    
                               
    
