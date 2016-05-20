# -*- coding: utf-8 -*-
"""
mix_functions.py

@author: Mengting Wan
"""
from __future__ import division

import numpy as np
import basic_functions as bsf 
import time as time
import KDEm as KDEm
import RKDE as RKDE

def compute_AUC(fpr, tpr):
    tmp = np.zeros((len(fpr),2))
    tmp[:,0] = fpr
    tmp[:,1] = tpr
    tmp.sort(axis=0)
    auc = tmp[0,0]*tmp[0,1]/2
    for i in range(1,len(tpr)):
        auc = auc + (tmp[i,0]-tmp[i-1,0])*(tmp[i,1]+tmp[i-1,1])/2
    auc = auc + (1-tmp[i,0])*(1+tmp[i,0])/2
    return(auc)

def twist_AUC(x, x_i, wi_vec, argmax=False, h=-1, tol=1e-3):
    l = len(x)
    center = np.array([x[0]])
    if(h<0):
        h = bsf.MAD(x_i)
        #h = np.std(x_i)
    conf = np.array([bsf.get_density(x[0],x_i,wi_vec,h)])
    ind = np.zeros(l)
    if(sum(wi_vec)==0):
        wi_vec = wi_vec + 1e-5
    for i in range(1,l):
        tmp = abs((x[i]-center)/center)
        if(tmp.min()>tol):
            center = np.append(center, x[i])
            tmp =  bsf.get_density(x[i],x_i,wi_vec,h)
            conf = np.append(conf, tmp)
            ind[i] = len(center)-1
        else:
            ind[i] = tmp.argmin()    
    conf = conf/sum(conf)
    thr = np.arange(0,1,0.05)
    center_list = []
    conf_list = []
    ind_list = []
    if(argmax):
        for i in range(len(center)):
            tmp0 = x_i[ind==i]
            tmp1 = np.zeros(len(tmp0))
            for j in range(len(tmp0)):
                tmp1[j] = bsf.get_density(tmp0[j],x_i,wi_vec,h)
            center[i] = tmp0[np.argmax(tmp1)]     
    for i in range(len(thr)):
        tmp = np.where(conf>thr[i])[0]
        center_new = center[list(tmp)]
        ind_new = -np.ones(l)
        for j in range(len(center_new)):
            ind_new[ind==tmp[j]] = j
        center_list.append(center_new)
        ind_list.append(np.copy(ind_new))
        tmp = np.copy(conf[conf>thr[i]])
        conf_list.append(tmp/sum(tmp))   
    return([center_list,np.array(ind_list),conf_list])
    
def evaluation(out1, out2, truth, c_max=2):
    l = len(out1)
    n_mark = len(truth)
    mae = []
    rmse = []
    truth_sort = []
    for j in range(n_mark):
        truth_sort.append(np.array(sorted(truth[j])))
    for k in range(c_max):
        mae.append(np.zeros((l,2)))
        rmse.append(np.zeros((l,2)))
    count = np.zeros(c_max)
    for i in range(l):
        for j in range(n_mark):
            if len(out1[i][j])==len(out2[i][j]) and len(out2[i][j])==len(truth[j]):
                k = len(truth[j])-1
                tmp1 = np.array(sorted(out1[i][j]))
                tmp2 = np.array(sorted(out2[i][j]))
                mae[k][i,0] += sum(abs(tmp1-truth_sort[j]))
                mae[k][i,1] += sum(abs(tmp2-truth_sort[j]))
                rmse[k][i,0] += sum((tmp1-truth_sort[j])**2)
                rmse[k][i,1] += sum((tmp2-truth_sort[j])**2)
                count[k] += k+1
    for k in range(c_max):
        mae[k] = mae[k]/count[k]
        rmse[k] = np.sqrt(rmse[k]/count[k])
    return([mae, rmse])                
        

def wKDE_twist(data, m, n, w_M, method, argmax=False, h=-1, p_unre=0.1):
    index, claim, count = bsf.extract(data, m, n)
    n = len(claim)
    truth = []
    ind_c = []
    conf = []
    evl = np.zeros((20,4))
    for i in range(n):        
        x_new = bsf.DENCLUE(claim[i], w_M[i], method, h=h)
        center,ind,conf_i = twist_AUC(x_new, claim[i], w_M[i], argmax, h)
        tmp0 = ind<0
        tmp1 = data[i][:,0]>=np.int(m*(1-p_unre))
        evl[:,0] = evl[:,0] + np.sum(tmp0*tmp1, axis=1)
        evl[:,1] = evl[:,1] + np.sum(tmp0*(tmp1==False), axis=1)
        evl[:,2] = evl[:,2] + np.sum((tmp0==False)*tmp1, axis=1)
        evl[:,3] = evl[:,3] + np.sum((tmp0==False)*(tmp1==False), axis=1)
        truth.append(center)
        ind_c.append(ind)
        conf.append(conf_i)
    tpr = evl[:,0]/(evl[:,0]+evl[:,2])
    fpr = evl[:,1]/(evl[:,1]+evl[:,3])
    auc = compute_AUC(fpr,tpr)
    rtn_truth = []
    rtn_ind = []
    rtn_conf = []
    for j in range(20):
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for i in range(n):
            tmp1.append(np.copy(truth[i][j]))
            tmp2.append(np.copy(ind_c[i][j]))
            tmp3.append(np.copy(conf[i][j]))
        rtn_truth.append(tmp1)
        rtn_ind.append(tmp2)
        rtn_conf.append(tmp3)
    return([rtn_truth,rtn_ind,rtn_conf, auc])

def KDE_twist(data, m, n, method, argmax=False, h=-1, p_unre=0.1):
    index, claim, count = bsf.extract(data, m, n)
    n = len(claim)
    truth = []
    ind_c = []
    conf = []
    evl = np.zeros((20,4))
    for i in range(n):        
        l = len(claim[i])
        x_new = bsf.DENCLUE(claim[i], np.ones(l)/l, method, h=h)
        center,ind,conf_i = twist_AUC(x_new, claim[i], np.ones(l)/l, argmax, h)
        tmp0 = ind<0
        tmp1 = data[i][:,0]>=np.int(m*(1-p_unre))
        evl[:,0] = evl[:,0] + np.sum(tmp0*tmp1, axis=1)
        evl[:,1] = evl[:,1] + np.sum(tmp0*(tmp1==False), axis=1)
        evl[:,2] = evl[:,2] + np.sum((tmp0==False)*tmp1, axis=1)
        evl[:,3] = evl[:,3] + np.sum((tmp0==False)*(tmp1==False), axis=1)
        truth.append(center)
        ind_c.append(ind)
        conf.append(conf_i)
    tpr = evl[:,0]/(evl[:,0]+evl[:,2])
    fpr = evl[:,1]/(evl[:,1]+evl[:,3])
    auc = compute_AUC(fpr,tpr)
    rtn_truth = []
    rtn_ind = []
    rtn_conf = []
    for j in range(20):
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for i in range(n):
            tmp1.append(np.copy(truth[i][j]))
            tmp2.append(np.copy(ind_c[i][j]))
            tmp3.append(np.copy(conf[i][j]))
        rtn_truth.append(tmp1)
        rtn_ind.append(tmp2)
        rtn_conf.append(tmp3)
    return([rtn_truth,rtn_ind,rtn_conf, auc])
    

def test_KDEm(data_raw, m, n, kernel, norm=True, argmax=False, max_itr=99, h=-1, p_unre=0.1):
    print "====Test with KDEm===="
    print "Kernel:", kernel
    if(norm):
        print "Normalized: True"
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        print "Normalized: False"
        data = data_raw[:]        
    a = time.time()
    source_score, weights_for_each, itr = KDEm.KDEm(data, m, n,method=kernel, h=h)
    b = time.time() - a
    print "Time cost for KDEm: "+str(b)+"s"
    out_list, ind_list, conf_list, auc = wKDE_twist(data, m, n, weights_for_each, kernel, argmax, h, p_unre)        
    c = time.time() - a
    print "Time cost for all: "+str(c)+"s"
    if(norm):        
        truth_out = []
        for i in range(20):
            truth_out.append(bsf.normalize_ivr(out_list[i], data_mean, data_sd))
    else:
        truth_out = out_list[:]
    print "====End test===="
    return([truth_out, ind_list, conf_list, source_score, auc, [b/itr,c]])


def test_KDE(data_raw, m, n, kernel, norm=True, argmax=False, h=-1, p_unre=0.1):
    print "====Test with KDE===="
    print "Kernel:", kernel
    if(norm):
        print "Normalized: True"
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        print "Normalized: False"
        data = data_raw[:]        
    a = time.time()    
    out_list, ind_list, conf_list, auc = KDE_twist(data, m, n, kernel, argmax, h, p_unre)        
    c = time.time() - a
    print "Time cost for all: "+str(c)+"s"
    if(norm):        
        truth_out = []
        for i in range(20):
            truth_out.append(bsf.normalize_ivr(out_list[i], data_mean, data_sd))
    else:
        truth_out = out_list[:]
    print "====End test===="
    return([truth_out, ind_list, conf_list, auc, [c]])

def test_RKDE(data_raw, m, n, kernel, rho_para=np.array([0.5,0.75,0.85]), norm=True, argmax=False, h=-1, p_unre=0.1):
    print "====Test with RKDE===="
    print "Kernel:", kernel
    if(norm):
        print "Normalized: True"
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        print "Normalized: False"
        data = data_raw[:]        
    a = time.time()
    weights_for_each=RKDE.RKDE(data, m, n, rho_para=rho_para, method=kernel)
    b = time.time() - a
    print "Time cost for RKDE: "+str(b)+"s"
    out_list, ind_list, conf_list, auc = wKDE_twist(data, m, n, weights_for_each, kernel, argmax, h, p_unre)        
    c = time.time() - a
    print "Time cost for all: "+str(c)+"s"
    if(norm):        
        truth_out = []
        for i in range(20):
            truth_out.append(bsf.normalize_ivr(out_list[i], data_mean, data_sd))
    else:
        truth_out = out_list[:]
    print "====End test===="
    return([truth_out, ind_list, conf_list, auc, [b,c]])
 
