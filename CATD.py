# -*- coding: utf-8 -*-
"""
CATD.py

@author: Mengting Wan
"""

from __future__ import division

import numpy as np
import basic_functions as bsf
import numpy.linalg as la
from scipy.stats import chi2

def update_w(claim, index, count, truth, m, n, eps=1e-15):
    rtn = -np.ones(m)
    for i in range(n):
        rtn[index[i]] = rtn[index[i]] + (claim[i]-truth[i])**2
    rtn[rtn==0] = 1e10
    rtn[rtn>0] = chi2.cdf(0.025, count[rtn>0])/rtn[rtn>0]
    #rtn[rtn>0] = chi2.interval(0.05, count[rtn>0])[0]/rtn[rtn>0]
    return(rtn)
    
def update_truth(claim, index, w_vec, m, n):
    rtn = np.zeros(n)
    for i in range(n):
        rtn[i] = np.dot(w_vec[index[i]],claim[i])/np.sum(w_vec[index[i]])
    return(rtn)

def CATD(data, m, n, intl=[], tol=0.1, max_itr=10):
    index, claim, count = bsf.extract(data, m, n)
    w_vec = np.ones(m)
    if(len(intl)>0):
        truth = update_truth(claim, index, w_vec, m, n)
    else:
        truth = np.copy(intl)
    err = 99
    itr = 0
    while(err > tol and itr < max_itr):
        w_old = np.copy(w_vec)
        w_vec = update_w(claim, index, count, truth, m, n)    
        truth = update_truth(claim, index, w_vec, m, n)
        err = la.norm(w_old-w_vec)/la.norm(w_old)
        itr = itr+1
    return([truth, w_vec])
    
def CATD_discret(data, m, n, intl=[]):
    index, claim, count = bsf.extract(data, m, n)
    w_vec = np.ones(m)
    if(len(intl)>0):
        truth = update_truth(claim, index, w_vec, m, n)
    else:
        truth = np.copy(intl)
    w_vec = update_w(claim, index, count, truth, m, n)    
    truth = np.zeros(n)
    for i in range(n):
        truth[i] = claim[i][w_vec[index[i]].argmax()]
    return([truth, w_vec])
