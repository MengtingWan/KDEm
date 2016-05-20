# -*- coding: utf-8 -*-
"""
KDEm.py

@author: Mengting Wan
"""

from __future__ import division

import numpy as np
import numpy.linalg as la
import basic_functions as bsf

# update source reliability scores            
def update_c(index, m, n, count, norm_M, method):
    rtn = np.zeros(m)
    for i in range(n):
        rtn[index[i]] = rtn[index[i]] + norm_M[i]/len(index[i])
    tmp = np.sum(rtn)
    if(tmp>0):
        rtn[rtn>0] = np.copy(-np.log((rtn[rtn>0]/count[rtn>0])/tmp))
    return([rtn,tmp])

# update opinion distributions
def update_w(index, m, n, c_vec, norm_M, method):
    w_M = []
    for i in range(n):
        w_i = np.zeros(len(index[i]))
        tmp = c_vec[index[i]]
        w_i[norm_M[i]>0] = tmp[norm_M[i]>0]
        tmp1 = sum(w_i)
        if(tmp1>0):
            w_M.append(w_i/tmp1)
        else:
            w_i[norm_M[i]==0] = 1
            tmp1 = sum(w_i)
            w_M.append(w_i/tmp1)
    return(w_M)

# implement KDEm without claim-value mappings
def KDEm(data, m, n, tol=1e-5, max_itr=99, method="Gaussian", h=-1):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    w_M = []
    for i in range(n):
        l = len(index[i])
        w_M.append(np.ones(l)/l)        
    itr=1
    kernel_M = bsf.get_kernel_matrix(claim, n, method)
    norm_M = bsf.get_norm_matrix(kernel_M, n, w_M, method)
    c_vec, J = update_c(index, m, n, count, norm_M, method)
    while((err > tol) & (itr < max_itr)):
        itr=itr+1
        J_old = J
        c_old = np.copy(c_vec)
        w_M = update_w(index, m, n, c_old, norm_M, method)
        norm_M = bsf.get_norm_matrix(kernel_M, n, w_M, method)
        c_vec, J = update_c(index, m, n, count, norm_M, method) 
        #err = la.norm(c_vec - c_old)/la.norm(c_old)
        err = abs((J-J_old)/J_old)
        #print itr,err
    print "#iteration:",itr
    return([c_vec, w_M, itr])

# implement KDEm with claim-value mappings
def KDEm_fast(data, m, n, tol=1e-5, max_itr=99, method="Gaussian", h=-1):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    data_c = bsf.compress(data)
    w_M = []
    for i in range(n):
        l = len(index[i])
        w_M.append(np.ones(l)/l)        
    itr=0
    kernel_M, value_M = bsf.get_kernel_matrix_fast(data_c, n, method)
    norm_M = bsf.get_norm_matrix_fast(data_c, kernel_M, value_M, w_M, method)
    c_vec, J = update_c(index, m, n, count, norm_M, method)
    while((err > tol) & (itr < max_itr)):
        itr=itr+1
        J_old = J
        c_old = np.copy(c_vec)
        w_M = update_w(index, m, n, c_old, norm_M, method)
        norm_M = bsf.get_norm_matrix_fast(data_c, kernel_M, value_M, w_M, method)
        c_vec ,J = update_c(index, m, n, count, norm_M, method)
        #err = la.norm(c_vec - c_old)/la.norm(c_old)
        err = abs((J-J_old)/J_old)
        #print itr,err
    print "#iteration:",itr
    return([c_vec, w_M, itr])
