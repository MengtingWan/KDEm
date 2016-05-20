# -*- coding: utf-8 -*-
"""
basic_functions.py

@author: Mengting Wan
"""
from __future__ import division

import numpy as np
import numpy.linalg as la


'''
#################################
Part1: Basic functions for Kernel Methods
#################################
'''

def K(x, method="Gaussian"):
    rtn = 0
    if(method.lower()=="uniform"):
        rtn = (abs(x)<=1)/2
    if(method.lower()=="epanechnikov" or method.lower()=="ep"):
        rtn = 3/4*(1-x**2)*(abs(x)<=1)
    if(method.lower()=="biweight" or method.lower()=="bi"):
        rtn = 15/16*(1-x**2)**2*(abs(x)<=1)
    if(method.lower()=="triweight" or method.lower()=="tri"):
        rtn = 35/32*(1-x**2)**3*(abs(x)<=1)
    if(method.lower()=="gaussian"):
        rtn = np.exp(-x**2)/np.sqrt(2*np.pi)
    if(method.lower()=="laplace"):
        rtn = np.exp(-abs(x))
    return(rtn)

def get_density(t,x_i,w_i,h):
    if(h>0):
        tmp = np.dot(w_i, K((t-x_i)/h))/(h*sum(w_i))
    else:
        tmp = 1
    return(tmp)

    
def get_kernel_matrix(claim, n, method, h=-1):
    kernel_M = []
    for i in range(n):
        x_i = claim[i]
        if(h<0):
            h = MAD(x_i)
            #h = np.std(x_i)
        l = x_i.shape[0]
        tmp = np.zeros((l,l))
        for j in range(l):
            if(h>0):
                tmp[j,:] = K((x_i[j]-x_i)/h, method)
            else:
                tmp[j,:] = K(0, method)
        kernel_M.append(tmp)
    return(kernel_M)

def get_norm_matrix(kernel_M, n, w_M, method):
    norm_M = []
    for i in range(n):
        kernel_m = kernel_M[i]
        term1 = np.diag(kernel_m)
        term2 = np.dot(kernel_m,w_M[i])
        term3 = np.dot(w_M[i],term2)
        tmp = term1-2*term2+term3
        tmp[tmp<0] = 0
        norm_M.append(tmp)
    return(norm_M)            


def get_kernel_matrix_fast(data_c, n, method, h=-1):
    kernel_M = []
    value_M = []
    for i in range(n):
        x_i = data_c[i].keys()
        if(h<0):
            h = MAD(decompress_i(data_c[i]))
            #h = np.std(decompress_i(data_c[i]))
        l = len(x_i)
        tmp = np.zeros((l,l))
        for j in range(l):
            if(h>0):
                tmp[j,:] = K((x_i[j]-x_i)/h, method)
            else:
                tmp[j,:] = K(0, method)
        kernel_M.append(tmp)
        value_M.append(x_i)
    return([kernel_M, value_M])

def get_norm_matrix_fast(data_c, kernel_M, value_M, w_M, method):
    norm_M = []
    n = len(data_c)
    for i in range(n):
        kernel_m = decompress_kernel(kernel_M[i], value_M[i], data_c[i], len(w_M[i]))
        term1 = np.diag(kernel_m)
        term2 = np.dot(kernel_m,w_M[i])
        term3 = np.dot(w_M[i],term2)
        tmp = term1-2*term2+term3
        tmp[tmp<0] = 0
        norm_M.append(tmp)
    return(norm_M)

'''
#################################
Part2: DENCLUE
#################################
'''
    
def DENCLUE(x_i, wi_vec, method="gaussian", tol=1e-8, h=-1):
                
    def cluster_update(x_old, x_i, wi_vec, h, method):
        l = len(wi_vec)
        tmp0 = np.ones((l,l))
        tmp1 = np.ones((l,l))
        if(method.lower()=="epanechnikov" or method.lower()=="ep"):
            for i in range(l):
                tmp0[:,i] = K((x_old[i]-x_i)/h, method="uniform")
                tmp1[:,i] = tmp0[:,i]*x_i
        if(method.lower()=="biweight" or method.lower()=="bi"):
            for i in range(l):
                tmp0[:,i] = K((x_old[i]-x_i)/h, method="ep")
                tmp1[:,i] = tmp0[:,i]*x_i
        if(method.lower()=="triweight" or method.lower()=="tri"):
            for i in range(l):
                tmp0[:,i] = K((x_old[i]-x_i)/h, method="bi")
                tmp1[:,i] = tmp0[:,i]*x_i
        if(method.lower()=="gaussian"):
            for i in range(l):
                tmp0[:,i] = K((x_old[i]-x_i)/h)
                tmp1[:,i] = tmp0[:,i]*x_i
        rtn = np.dot(wi_vec,tmp1)
        rtn_de = np.dot(wi_vec,tmp0)
        rtn[rtn_de>0] = rtn[rtn_de>0]/rtn_de[rtn_de>0]
        return(rtn)
        
    err = 99
    if(h<0):
        h = MAD(x_i)
        #h = np.std(x_i)
    if(sum(wi_vec)==0):
        wi_vec = wi_vec + 1e-5
    if(np.var(x_i)>0):
        x_new = np.copy(x_i) + 1e-12
        while(err > tol):
            x_old = np.copy(x_new)
            x_new = cluster_update(x_old, x_i, wi_vec, h=h, method=method)
            err = la.norm(x_old-x_new)/la.norm(x_old)
    else:
        x_new = np.copy(x_i)
    return(x_new)    

def twist(x, x_i, wi_vec, argmax=False, cut=0, h=-1, tol=1e-3):
    l = len(x)
    center = np.array([x[0]])
    if(h<0):
        h = MAD(x_i)
        #h = np.std(x_i)
    conf = np.array([get_density(x[0],x_i,wi_vec,h)])
    ind = np.zeros(l)
    if(sum(wi_vec)==0):
        wi_vec = wi_vec + 1e-5
    for i in range(1,l):
        tmp = abs((x[i]-center)/center)
        if(tmp.min()>tol):
            center = np.append(center, x[i])
            conf = np.append(conf, get_density(x[i],x_i,wi_vec,h))
            ind[i] = len(center)-1
        else:
            ind[i] = tmp.argmin()    
    conf = conf/sum(conf)
    if(cut>0):
        tmp = np.where(conf>cut)[0]
        center_new = center[list(tmp)]
        ind_new = -np.ones(l)
        for i in range(len(center_new)):
            ind_new[ind==tmp[i]] = i
        center = np.copy(center_new)
        ind = np.copy(ind_new)
        conf = conf[conf>cut]
        conf = conf/sum(conf)
    
    if(argmax):
        for i in range(len(center)):
            tmp0 = x_i[ind==i]
            tmp1 = np.zeros(len(tmp0))
            for j in range(len(tmp0)):
                tmp1[j] = get_density(tmp0[j],x_i,wi_vec,h)
            center[i] = tmp0[np.argmax(tmp1)]        
    return([center,ind,conf])

def wKDE_twist(data, m, n, w_M, method, argmax, cut=0, h=-1):
    index, claim, count = extract(data, m, n)
    n = len(claim)
    truth = []
    ind_c = []
    conf = []
    for i in range(n):        
        x_new = DENCLUE(claim[i], w_M[i], method, h=h)
        center,ind,conf_i = twist(x_new, claim[i], w_M[i], argmax, cut, h=h)
        truth.append(center)
        ind_c.append(ind)
        conf.append(conf_i)
    return([truth, ind_c, conf])

def KDE_twist(data, m, n, method, argmax, cut=0, h=-1):
    index, claim, count = extract(data, m, n)
    n = len(claim)
    truth = []
    ind_c = []
    conf = []
    for i in range(n):        
        l = len(claim[i])
        x_new = DENCLUE(claim[i], np.ones(l)/l, method, h=h)
        center,ind,conf_i = twist(x_new, claim[i], np.ones(l)/l, argmax, cut, h=h)
        truth.append(center)
        ind_c.append(ind)
        conf.append(conf_i)
    return([truth, ind_c, conf])

'''
#################################
Part3: Others
#################################
'''


def normalize(data):
    data_new = []
    n = len(data)
    data_mean = np.zeros(n)
    data_sd = np.zeros(n)
    for i in range(n):
        data_i = np.float64(np.copy(data[i]))
        data_mean[i] = np.mean(data[i][:,1])
        data_sd[i] = np.std(data[i][:,1])
        if(data_sd[i]>0):
            data_i[:,1] = (data[i][:,1] - data_mean[i])/data_sd[i]
        else:
            data_i[:,1] = (data[i][:,1] - data_mean[i])
        data_new.append(data_i)
    return([data_new,data_mean,data_sd])

def normalize_ivr(data, data_mean,data_sd):
    data_new = []
    n = len(data)
    for i in range(n):
        data_i = np.copy(data[i])*data_sd[i]+data_mean[i]        
        data_new.append(data_i)
    return(data_new)

def extract(data, m, n):
    index=[]
    claim=[]
    count = np.zeros(m)
    for i in range(n):
        src = list(data[i][:,0])
        count[src] = count[src] + 1
        index.append(src)
        claim.append(data[i][:,1])
    return([index,claim,count])

def compress_i(datai):
    xi = {}
    l = len(datai)
    for j in range(l):
        #src = datai[j,0]
        val = datai[j][1:]
        if(xi.has_key(val)):
            tmp = xi[val][:]
            tmp.append(j)
            xi[val] = tmp
        else:
            xi[val] = [j]
    return(xi)

def compress(data):
    n = len(data)
    data_c = []
    for i in range(n):
        datai = {}
        l = len(data[i])
        for j in range(l):
            #src = data[i][j,0]            
            val = data[i][j][1]
            if(datai.has_key(val)):
                tmp = datai[val][:]
                tmp.append(j)
                datai[val] = tmp
            else:
                datai[val] = [j]
        data_c.append(datai)
    return(data_c)

def decompress_i(datai):
    xi = []
    for (k,v) in datai.items():
        for j in range(len(v)):
            xi.append(k)
    return(np.array(xi))

def decompress_kernel(kernel_c, valuei, datai, ni):
    kernel_m = np.zeros((ni,ni))
    l = len(valuei)
    for i in range(l):
        for j in range(l):
            for v in datai[valuei[j]]:
                kernel_m[datai[valuei[i]],v] = kernel_c[i,j]
    return(kernel_m)
        
def get_moments(x_i, wi_vec, h, method="gaussian"):
    if(h<0):
        h = MAD(x_i)
        #h = np.std(x_i)
    mu = np.dot(wi_vec, x_i)
    if(method=="laplace"):
        m2 = np.dot(wi_vec, (x_i-mu)**2+2*h**2)
        m3 = np.dot(wi_vec, (x_i-mu)**3)
        m4 = np.dot(wi_vec, (x_i-mu)**4+6*(x_i-mu)**2*2*h**2+24*h**4)
    else:
        m2 = np.dot(wi_vec, (x_i-mu)**2+h**2)
        m3 = np.dot(wi_vec, (x_i-mu)**3)
        m4 = np.dot(wi_vec, (x_i-mu)**4+6*(x_i-mu)**2*h**2+3*h**4)
    skewness = m3/(m2)**(3/2)
    kurtosis = m4/(m2)**2 - 3
    stats = skewness*2 - kurtosis
    return([skewness, kurtosis, stats])

def MAD(x_i):
    return(np.median(abs(x_i-np.median(x_i,axis=0)),axis=0)+1e-10*np.std(x_i,axis=0))
