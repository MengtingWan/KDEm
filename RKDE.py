# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:23:20 2016

@author: mwan
"""
from __future__ import division

import numpy as np
import numpy.linalg as la
import basic_functions as bsf

    
def compute_a(norm_M, n, para=np.array([0.5,0.75,0.85])):
    a_vec = np.zeros((n,3))
    for i in range(n):
        tmp = sorted(np.sqrt(norm_M[i]))
        for j in range(3):
            if(para[j]>0):
                a_vec[i,j] = tmp[np.int(len(tmp)*para[j])]
            else:
                a_vec[i,j] = tmp[-1] + 0.1
    return(a_vec)
    
    
def loss(x, method, a_vec=np.zeros(3)):
    rtn = 0
    if(sum(a_vec)==0):
        tmp = sorted(x)
        a_vec[0] = tmp[np.int(len(tmp)*0.5)]
        a_vec[1] = tmp[np.int(len(tmp)*0.75)]
        a_vec[2] = tmp[np.int(len(tmp)*0.85)]
    a = a_vec[0]
    b = a_vec[1]
    c = a_vec[2]
    if(sum(x<0)==0):
        rtn = np.zeros(len(x))
        
        rtn[x<=a] = x[x<=a]**2/2
        rtn[(x>a)*(x<=b)] = a*x[(x>a)*(x<=b)]-a*a/2
        rtn[(x>b)*(x<=c)] = a*(x[(x>b)*(x<=c)]-c)**2/(2*(b-c))+a*(b+c-a)/2
        rtn[x>c] = a*(b+c-a)/2
        
        '''
        tmp = x[x<=c]/c
        rtn[x<=c] = 1-(1-tmp**2)**3
        rtn[x>c] = 1
        '''
        #rtn = a**2*np.log(1+(x/a)**2)
        #rtn = a**2*(np.sqrt(1+(x/a)**2)-1)
    return(rtn)

def loss_dev(x, method, a_vec=np.zeros(3)):
    rtn = 0
    if(sum(a_vec)==0):
        tmp = sorted(x)
        a_vec[0] = tmp[np.int(len(tmp)*0.5)]
        a_vec[1] = tmp[np.int(len(tmp)*0.75)]
        a_vec[2] = tmp[np.int(len(tmp)*0.85)]
    a = a_vec[0]
    b = a_vec[1]
    c = a_vec[2]
    if(sum(x<0)==0):
        rtn = np.zeros(len(x))
        
        rtn[x<=a] = x[x<=a]
        rtn[(x>a)*(x<=b)] = a
        rtn[(x>b)*(x<=c)] = a*(x[(x>b)*(x<=c)]-c)/(b-c)
        rtn[x>c] = 0
        
        '''
        tmp = x[x<=c]/c
        rtn[x<=c] = 6*tmp/c*(1-tmp**2)**2
        '''
        #rtn = 2*x/(1+(x/a)**2)
        #rtn = 2*x/np.sqrt(1+(x/a)**2)
    return(rtn)

def update_w_RKDE(index, m, n, c_vec, norm_M, a_vec, method):
    w_M = []
    for i in range(n):
        w_i = np.zeros(len(index[i]))
        tmp = c_vec[index[i]] * loss_dev(np.sqrt(norm_M[i]), method, a_vec[i,:])/np.sqrt(norm_M[i])
        w_i[norm_M[i]>0] = tmp[norm_M[i]>0]
        tmp1 = sum(w_i)
        if(tmp1>0):
            w_M.append(w_i/tmp1)
        else:
            w_i[norm_M[i]==0] = 1
            tmp1 = sum(w_i)
            if(tmp1>0):
                w_M.append(w_i/tmp1)
            else:
                w_M.append(np.ones(len(w_i))/len(w_i))
    return(w_M)

def update_c_RKDE(index, m, n, count, norm_M, a_vec, method):
    rtn = np.zeros(m)
    for i in range(n):
        rtn[index[i]] = rtn[index[i]] + loss(np.sqrt(norm_M[i]), method, a_vec[i,:])/len(index[i])
    #rtn[rtn>0] = rtn[rtn>0]/count[rtn>0]
    tmp = np.sum(rtn)
    if(tmp>0):
        rtn[rtn>0] = np.copy(-np.log((rtn[rtn>0]/count[rtn>0])/tmp))
    return([rtn,tmp])        
    
def RKDE(data, m, n, rho_para=np.array([0.5,0.75,0.85]), tol=1e-3, max_itr=99, method="Gaussian"):
    index,claim,count = bsf.extract(data, m, n)
    w_M = []
    for i in range(n):
        l = len(index[i])
        w_M.append(np.ones(l)/l)        
    kernel_M = bsf.get_kernel_matrix(claim, n, method)
    norm_M = bsf.get_norm_matrix(kernel_M, n, w_M, method)
    c_vec = np.ones(m)
    a_vec = compute_a(norm_M, n, rho_para)
    w_M = []
    for i in range(n):
        kernel_m = np.copy(kernel_M[i])
        w_i, norm = RKDE_single(index[i], m, n, c_vec, kernel_m, a_vec[i,:], method, max_itr=max_itr, tol=1e-3)
        w_M.append(w_i)
    return(w_M)

def RKDE_single(index, m, n, c_vec, kernel_m, a_vec, method, max_itr=50, tol=1e-2):
    err = 99
    itr = 1
    w_i = np.ones(len(index))/len(index)
    term1 = np.diag(kernel_m)
    term2 = np.dot(kernel_m,w_i)
    term3 = np.dot(w_i,term2)
    norm = term1-2*term2+term3
    norm[norm<0] = 0
    while((err > tol) & (itr < max_itr)):
        itr=itr+1
        w_old = np.copy(w_i)
        w_i = np.zeros(len(index))
        tmp = c_vec[index]*loss_dev(np.sqrt(norm), method, a_vec)/np.sqrt(norm)
        w_i[norm>0] = tmp[norm>0]
        tmp1 = sum(w_i)
        if(tmp1>0):
            w_i = w_i/tmp1
        else:
            w_i[norm==0] = 1
            tmp1 =sum(w_i)
            if(tmp1>0):
                w_i = w_i/tmp1
            else:
                w_i = np.ones(len(w_i))/len(w_i)
        term2 = np.dot(kernel_m,w_i)
        term3 = np.dot(w_i,term2)
        norm = term1-2*term2+term3
        norm[norm<0] = 0
        err = la.norm(w_i-w_old)/la.norm(w_old)
    #print itr, err
    return([w_i, norm])
