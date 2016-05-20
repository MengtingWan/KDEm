# -*- coding: utf-8 -*-
"""
CRH.py

@author: Mengting Wan
"""

from __future__ import division

import numpy as np
import numpy.linalg as la
import basic_functions as bsf

def update_w(claim, index, truth, m, n, eps=1e-15):
    rtn = np.zeros(m)
    for i in range(n):
        rtn[index[i]] = rtn[index[i]] + (claim[i]-truth[i])**2/max(np.std(claim[i]),eps)
    tmp = np.sum(rtn)
    if(tmp>0):
        rtn[rtn>0] = np.copy(-np.log(rtn[rtn>0]/tmp))
    return(rtn)
    
def update_truth(claim, index, w_vec, m, n):
    rtn = np.zeros(n)
    for i in range(n):
        rtn[i] = np.dot(w_vec[index[i]],claim[i])/np.sum(w_vec[index[i]])
    return(rtn)

def CRH(data, m, n, tol=1e-3, max_itr=99):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    w_vec = np.ones(m)
    truth = np.zeros(n)
    while((err > tol) & (itr < max_itr)):
        itr = itr+1
        truth_old = np.copy(truth)
        truth = update_truth(claim, index, w_vec, m, n)
        w_vec = update_w(claim, index, truth, m, n)
        err = la.norm(truth-truth_old)/la.norm(truth_old)
    return([truth, w_vec])
    
def CRH_discret(data, m, n, tol=1e-3, max_itr=99):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    w_vec = np.ones(m)
    truth = np.zeros(n)
    while((err > tol) & (itr < max_itr)):
        itr = itr+1
        truth_old = np.copy(truth)
        truth = update_truth(claim, index, w_vec, m, n)
        w_vec = update_w(claim, index, truth, m, n)
        err = la.norm(truth-truth_old)/la.norm(truth_old)
    truth = np.zeros(n)
    for i in range(n):
        truth[i] = claim[i][w_vec[index[i]].argmax()]
    return([truth, w_vec])