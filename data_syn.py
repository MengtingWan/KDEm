# -*- coding: utf-8 -*-
"""
data_syn.py

@author: Mengting Wan
"""

from __future__ import division

import numpy as np
#import scipy.stats as spst

def data_unimodal(m, n, lbda=5, p_unre=0):
    truth = list(np.ones(n).reshape(n,1))
    conf = list(np.ones(n).reshape(n,1))
    tmp = np.ones((n,2))*2
    tmp[:,0] = np.random.poisson(lam=lbda,size=n)
    ni = np.max(tmp,axis=1)
    
    m1 = np.int(m*(1-p_unre))
    sigma = np.random.rand(m)*0.04+0.01
    sigma[m1:] = np.random.rand(m-m1)*4+1
    
    data = []
    for i in range(n):
        index = np.random.permutation(m)[:ni[i]]
        datai = np.ones((ni[i],3))
        datai[:,0] = index
        tmp = sigma[list(index)]
        datai[:,1] = np.random.normal(loc=1,scale=tmp)
        data.append(datai)
    return([data, truth, conf])

def data_bimodal(m, n, lbda=10, p_unre=0):
    truth = np.ones((n,2))
    truth[:,1] = np.random.normal(loc=1, scale=10, size=n)
    conf = np.ones((n,2))/2
    tmp = np.ones((n,2))*2
    tmp[:,0] = np.random.poisson(lam=lbda,size=n)
    ni = np.max(tmp,axis=1)
    
    m1 = np.int(m*(1-p_unre))
    sigma = np.ones(m)*0.04+0.01
    sigma[m1:] = np.random.rand(m-m1)*4+1
    
    data = []
    for i in range(n):
        index = np.random.permutation(m)[:ni[i]]
        datai = np.ones((ni[i],3))
        datai[:,0] = index
        tmp = np.random.randint(2, size=ni[i])
        datai[:,2] = tmp
        tmp1 = sigma[list(index)]
        datai[:,1] = np.random.normal(loc=truth[i][list(tmp)],scale=tmp1)
        data.append(datai)
    return([data, truth, conf])

def data_mix(m, n, lbda1=5, lbda2=10, p_unre1=0, p_unre2=0):
    n1 = np.int(n/2)
    n2 = n-n1
    data1, truth1, conf1 = data_unimodal(m, n1, lbda=lbda1, p_unre=p_unre1)
    data2, truth2, conf2 = data_bimodal(m, n2, lbda=lbda2, p_unre=p_unre2)
    data = []
    truth = []
    conf = []
    for i in range(n1):
        data.append(np.copy(data1[i]))
        truth.append(np.array(truth1[i]))
        conf.append(np.array(conf1[i]))
    for i in range(n2):
        data.append(np.copy(data2[i]))
        truth.append(np.array(truth2[i]))
        conf.append(np.array(conf2[i]))
    return([data, truth, conf])


def reload_data(path="syn_data.txt"):
    fileHandle = open(path)
    f = fileHandle.readline()
    data = []
    truth = []
    conf = []
    record = []
    while(f!=''):
        dataline = f.strip()
        if f[0] == 'o':
            if(len(record)>0):
                data.append(np.array(record))
            record = []
            tmp = dataline.split('\t')
            item = tmp[1].split(' ')
            truth_i = []
            conf_i = []
            for i in range(len(item)):
                tmp1 = item[i].split(':')
                truth_i.append(np.float64(tmp1[0]))
                conf_i.append(np.float64(tmp1[1]))
            truth.append(np.array(truth_i))
            conf.append(np.array(conf_i))
        else:
            item = dataline.split(' ')
            record.append([np.float64(item[0]),np.float64(item[1]),np.float64(item[2])])
        f = fileHandle.readline()
    data.append(np.array(record))
    return([data,truth,conf])

def save_syn(data_raw, truth0, conf0, path="syn_data.txt"):
    n = len(data_raw)
    out = open(path,'w')
    for i in range(n):
        print>>out, 'object '+str(i)+'\t',
        for j in range(len(truth0[i])):
            print>>out,str(truth0[i][j])+':'+str(conf0[i][j]),
        print>>out
        k = data_raw[i].shape[0]
        for j in range(k):
            print>>out,data_raw[i][j,0],data_raw[i][j,1],data_raw[i][j,2]
    out.close()
