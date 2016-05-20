# -*- coding: utf-8 -*-
"""
TruthFinder.py

@author: Mengting Wan
"""


from __future__ import division

import numpy as np
import numpy.linalg as la
import basic_functions as bsf


def update_source(claim, index, s_set, m, n):
    t_vec = np.zeros(m)
    tau_vec = np.zeros(m)
    count = np.zeros(m)
    for i in range(n):
        t_vec[index[i]] = t_vec[index[i]] + s_set[i]
        count[index[i]] = count[index[i]] + 1
    t_vec[count>0] = t_vec[count>0]/count[count>0]
    tau_vec[t_vec>=1] = np.log(1e10)
    tau_vec[t_vec<1] = -np.log(1-t_vec[t_vec<1])
    return(tau_vec)
    
def update_claim(claim, index, tau_vec, m, n, rho, gamma, base_thr=0):
    s_set= []
    for i in range(n):
        claim_set = list(set(claim[i]))
        sigma_i = np.zeros(len(claim_set))
        s_vec = np.zeros(len(claim[i]))
        for j in range(len(claim_set)):
            sigma_i[j] = sum(tau_vec[index[i]][claim[i]==claim_set[j]])
        tmp_i = np.copy(sigma_i)
        for j in range(len(claim_set)):
            tmp_i[j] = (1-rho*(1-base_thr))*sigma_i[j] + rho*sum((np.exp(-abs(claim_set-claim_set[j]))-base_thr)*sigma_i)
            #tmp_i[j] = (1+rho)*sigma_i[j] + rho*sum(-sigma_i)
            s_vec[claim[i]==claim_set[j]] = 1/(1 + np.exp(-gamma*tmp_i[j]))
        s_set.append(s_vec)
    return(s_set)

def TruthFinder(data, m, n, tol=0.1, max_itr=10):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    tau_vec = -np.log(1-np.ones(m)*0.9)
    truth = np.zeros(n)
    rho = 0.5
    gamma = 0.3
    while((err > tol) & (itr < max_itr)):
        itr = itr+1
        tau_old = np.copy(tau_vec)
        s_set = update_claim(claim, index, tau_vec, m, n, rho, gamma)
        tau_vec = update_source(claim, index, s_set, m, n)
        err = 1 - np.dot(tau_vec,tau_old)/(la.norm(tau_vec)*la.norm(tau_old))
        print itr, err
    truth = np.zeros(n)
    for i in range(n):
        truth[i] = claim[i][np.argmax(s_set[i])]
    return([truth, tau_vec])