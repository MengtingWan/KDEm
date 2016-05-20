# -*- coding: utf-8 -*-
"""
GTM.py

@author: Mengting Wan
"""
from __future__ import division

import numpy as np
import numpy.linalg as la
import basic_functions as bsf
#import TruthFinder as TruthFinder

def E_step(claim, index, m, n, sigma_vec, mu0, sigma0):
    truth = np.zeros(n)
    for i in range(n):
        tmp = mu0/sigma0**2 + sum(claim[i]/sigma_vec[index[i]]**2)
        tmp1 = 1/sigma0**2 + sum(1/sigma_vec[index[i]]**2)
        truth[i] = tmp/tmp1
    return(truth)

def M_step(claim, index, m, n, truth, alpha, beta):
    sigma_vec = np.zeros(m)
    count = np.zeros(m)
    for i in range(n):
        sigma_vec[index[i]] = sigma_vec[index[i]] + 2*beta + (claim[i] - truth[i])**2
        count[index[i]] = count[index[i]] + 1
    sigma_vec = sigma_vec / (2*(alpha+1)+count)
    return(sigma_vec)

def Initialization(intl, claim, index, m, n, alpha, beta):
    if(len(intl)>0):
        truth = np.zeros(n)
        for i in range(n):
            truth[i] = np.median(claim[i])      
    else:
        truth = np.copy(intl)
    sigma_vec = M_step(claim, index, m, n, truth, alpha, beta)
    return([truth, sigma_vec])
    
def GTM(data, m, n, intl=[], tol=1e-3, max_itr=99, alpha=10, beta=10, mu0=0, sigma0=1):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    truth, sigma_vec = Initialization(intl, claim, index, m, n, alpha, beta) 
    #truth, tau = TruthFinder.TruthFinder(data, m, n)
    while((err > tol) & (itr < max_itr)):
        itr = itr+1
        truth_old = np.copy(truth)
        truth = E_step(claim, index, m, n, sigma_vec, mu0, sigma0)
        sigma_vec = M_step(claim, index, m, n, truth, alpha, beta)
        err = la.norm(truth-truth_old)/la.norm(truth_old)
    return([truth, sigma_vec])
    
def GTM_discret(data, m, n, intl=[], tol=1e-3, max_itr=99, alpha=10, beta=10, mu0=0, sigma0=1):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    truth, sigma_vec = Initialization(intl, claim, index, m, n, alpha, beta) 
    #truth, tau = TruthFinder.TruthFinder(data, m, n)
    while((err > tol) & (itr < max_itr)):
        itr = itr+1
        truth_old = np.copy(truth)
        truth = E_step(claim, index, m, n, sigma_vec, mu0, sigma0)
        sigma_vec = M_step(claim, index, m, n, truth, alpha, beta)
        err = la.norm(truth-truth_old)/la.norm(truth_old)
    truth = np.zeros(n)
    for i in range(n):
        truth[i] = claim[i][sigma_vec[index[i]].argmin()]
    return([truth, sigma_vec])    