# -*- coding: utf-8 -*-
"""
test_function.py

@author: Mengting Wan
"""
from __future__ import division

import numpy as np
from collections import Counter
import basic_functions as bsf
import KDEm as KDEm
import RKDE as RKDE
import time
import CRH as CRH
import TruthFinder as TruthFinder
import GTM as GTM
import Accu as Accu
import CATD as CATD


def test_basic(data_raw, m, n, tp="voting", norm=True):
    if(norm):
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        data = data_raw[:]
    truth_set = np.zeros((n,8))
    print "Test with Baseline Methods..."
    for i in range(n):
        if(len(data[i])>0):
            #==== Mean ====
            truth_set[i,0] = np.mean(data[i][:,1])
            #==== Median ====
            truth_set[i,1] = np.median(data[i][:,1])
            #==== Voting/Maximal ===
            if(tp=="voting"):
                c = Counter(data[i][:,1])
                truth_set[i,2] = c.most_common(1)[0][0]        
            if(tp=="maximal"):
                tmp = np.histogram(data[i][:,1], bins=data[i].shape[0])
                truth_set[i,2] = (tmp[1][np.argmax(tmp[0])]+tmp[1][np.argmax(tmp[0])+1])/2
        else:
            truth_set[i,:] = np.nan    
    #==== TruthFinder ====
    truth_set[:,3],tau_vec = TruthFinder.TruthFinder(data, m, n)
    #==== AccuSim & AccuCopy====
    truth_set[:,4] =  Accu.AccuSim(data, m, n)
    #==== GTM ====
    truth_set[:,5], sigma_vec = GTM.GTM(data, m, n, intl=truth_set[:,3])
    #==== CRH ====
    truth_set[:,6],w_vec = CRH.CRH(data, m, n)
    #==== CATD ====
    truth_set[:,7],w_vec = CATD.CATD(data, m, n, intl=truth_set[:,3])
    
    if(norm):        
        for i in range(truth_set.shape[1]):
            truth_set[:,i] = bsf.normalize_ivr(truth_set[:,i], data_mean, data_sd)
    print "End test."
    return(truth_set)
    

def get_moments(data, m, n, w_M, method="gaussian", h=-1):
    moments = np.zeros((n,3))
    for i in range(n):
        x_i = np.copy(data[i][:,1])
        if(len(w_M)>0):
            moments[i,:] = bsf.get_moments(x_i, w_M[i], h)
        else:
            moments[i,:] = bsf.get_moments(x_i, np.ones(len(x_i))/len(x_i), h)
    return(moments)


def test_KDEm(data_raw, m, n, kernel, norm=True, outlier_thr=0, max_itr=99, argmax=False, h=-1):
    print "Test with KDEm..."
    print "Kernel:", kernel
    if(norm):
        print "Normalized: True"
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        print "Normalized: False"
        data = data_raw[:]        
    a = time.time()
    source_score, weights_for_each, itr = KDEm.KDEm(data, m, n, max_itr=max_itr, method=kernel, h=h)
    b = time.time() - a
    print "Time cost for each iteration in KDEm: "+str(b)+"s"
    out, cluster_index, cluster_confidence = bsf.wKDE_twist(data, m, n, weights_for_each, kernel, argmax, outlier_thr, h)        
    c = time.time() - a
    #print "Time cost for all: "+str(c)+"s"
    moments = get_moments(data, m, n, weights_for_each, method=kernel, h=h)
    if(norm):        
        truth_out = bsf.normalize_ivr(out, data_mean, data_sd)
    else:
        truth_out = out[:]
    print "End."
    return([truth_out, cluster_index, cluster_confidence, source_score, weights_for_each, moments, [b/itr,c]])


def test_KDEm_fast(data_raw, m, n, kernel, norm=True, outlier_thr=0, max_itr=99, argmax=False, h=-1):
    print "Test with KDEm..."
    print "Kernel:", kernel
    if(norm):
        print "Normalized: True"
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        print "Normalized: False"
        data = data_raw[:]        
    a = time.time()
    source_score, weights_for_each, itr = KDEm.KDEm_fast(data, m, n, max_itr=max_itr, method=kernel, h=h)
    b = time.time() - a
    print "Time cost for each iteration in KDEm: "+str(b)+"s"
    out, cluster_index, cluster_confidence = bsf.wKDE_twist(data, m, n, weights_for_each, kernel, argmax, outlier_thr, h=h)        
    c = time.time() - a
    #print "Time cost for all: "+str(c)+"s"
    moments = get_moments(data, m, n, weights_for_each, method=kernel, h=h)
    if(norm):        
        truth_out = bsf.normalize_ivr(out, data_mean, data_sd)
    else:
        truth_out = out[:]
    print "End."
    return([truth_out, cluster_index, cluster_confidence, source_score, weights_for_each, moments, [b/itr,c]])


def test_KDE(data_raw, m, n, kernel, norm=True, outlier_thr=0, argmax=False, h=-1):
    print "Test with KDE..."
    print "Kernel:", kernel
    if(norm):
        print "Normalized: True"
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        print "Normalized: False"
        data = data_raw[:]        
    a = time.time()
    out, cluster_index, cluster_confidence = bsf.KDE_twist(data, m, n, kernel, argmax, outlier_thr, h=h)     
    c = time.time() - a
    #print "Time cost for all: "+str(c)+"s"
    moments = get_moments(data, m, n, w_M=[], method=kernel, h=h)
    if(norm):        
        truth_out = bsf.normalize_ivr(out, data_mean, data_sd)
    else:
        truth_out = out[:]
    print "End."
    return([truth_out, cluster_index, cluster_confidence, moments, [c]])
    
def test_RKDE(data_raw, m, n, kernel, rho_para=np.array([0.5,0.75,0.85]), norm=True, time_report=True, outlier_thr=0.05, max_itr=30, argmax=False):
    print "====Test with RKDE===="
    print "Kernel:", kernel
    if(time_report):
        if(norm):
            print "Normalized: True"
            data, data_mean, data_sd = bsf.normalize(data_raw)
        else:
            print "Normalized: False"
            data = data_raw[:]        
        a = time.time()
        weights_for_each=RKDE.RKDE(data, m, n, rho_para=rho_para, max_itr=max_itr, method=kernel)
        b = time.time() - a
        print "Time cost for RKDE: "+str(b)+"s"
        out, cluster_index, cluster_confidence = bsf.wKDE_twist(data, m, n, weights_for_each, kernel, argmax, cut=outlier_thr)        
        c = time.time() - a
        print "Time cost for all: "+str(c)+"s"
        moments = get_moments(data, m, n, weights_for_each, method=kernel)
        if(norm):        
            truth_out = bsf.normalize_ivr(out, data_mean, data_sd)
        else:
            truth_out = out[:]
        print "====End test===="
        return([truth_out, cluster_index, cluster_confidence, weights_for_each, moments, [b,c]])
    else:
        if(norm):
            print "Normalized: True"
            data, data_mean, data_sd = bsf.normalize(data_raw)
        else:
            print "Normalized: False"
            data = data_raw[:]        
        weights_for_each=RKDE.RKDE(data, m, n, rho_para=rho_para, max_itr=max_itr, method=kernel)
        out, cluster_index, cluster_confidence = bsf.wKDE_twist(data, m, n, weights_for_each, kernel, argmax, cut=outlier_thr)        
        moments = get_moments(data, m, n, weights_for_each, method=kernel)        
        if(norm):        
            truth_out = bsf.normalize_ivr(out, data_mean, data_sd)
        else:
            truth_out = out[:]
        print "====End test===="
        return([truth_out, cluster_index, cluster_confidence, weights_for_each, moments])
