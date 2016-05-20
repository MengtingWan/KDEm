# -*- coding: utf-8 -*-
"""
test_syn_AUC.py

@author: Mengting Wan
"""
from __future__ import division

import numpy as np
import data_syn as syn
import evaluation
import mix_functions as mix

def run_syn_mix(m=200, n=100, load="", save="", lbda1=10, lbda2=10, p_unre1=0, p_unre2=0):
    if(load!=""):
        data_raw, truth, truth_conf = syn.reload_data(load)
    else:
        data_raw, truth, truth_conf = syn.data_mix(m, n, lbda1, lbda2, p_unre1, p_unre2)
    measure_all = []
    size = np.zeros(2)
    for i in range(n):
        size = size + np.array([data_raw[i].shape[0],data_raw[i].shape[0]**2])
        
    kernel="gaussian"
    #Test with KDEm
    rtn = mix.test_KDEm(data_raw, m, n, kernel=kernel, norm=True, p_unre=p_unre1)
    out1, index1, conf1, source_score, auc1, time1 = rtn
    #Report test results
    fpr = np.zeros((2,len(out1)))
    tpr = np.zeros((2,len(out1)))
    for i in range(len(out1)):
        n_cluster_pre, evl_measure = evaluation.evaluation_basic(out1[i], conf1[i], truth, truth_conf)
        #evl_measure = TP,FP,FN,TN, precision,recall,F1,FPR, MAE,RMSE
        if(evl_measure.shape[0]<2):
            evl_measure = np.append(evl_measure, np.zeros((1,10)),axis=0)
        fpr[:,i] = evl_measure[:,7]
        tpr[:,i] = evl_measure[:,5]
    auc = np.zeros(3)
    auc[0] = mix.compute_AUC(fpr[0,:],tpr[0,:])
    auc[1] = mix.compute_AUC(fpr[1,:],tpr[1,:])
    auc[2] = auc1
    measure_all.append(auc)
    
    #Test with KDE        
    rtn = mix.test_KDE(data_raw, m, n, kernel=kernel, norm=True, p_unre=p_unre1)
    out2, cluster_index, conf2, auc2, time2 = rtn
    #Report test results
    fpr = np.zeros((2,len(out1)))
    tpr = np.zeros((2,len(out1)))
    for i in range(len(out1)):
        n_cluster_pre, evl_measure = evaluation.evaluation_basic(out2[i], conf2[i], truth, truth_conf)
        #evl_measure = TP,FP,FN,TN, precision,recall,F1,FPR, MAE,RMSE
        if(evl_measure.shape[0]<2):
            evl_measure = np.append(evl_measure, np.zeros((1,10)),axis=0)
        fpr[:,i] = evl_measure[:,7]
        tpr[:,i] = evl_measure[:,5]
    auc = np.zeros(3)
    auc[0] = mix.compute_AUC(fpr[0,:],tpr[0,:])
    auc[1] = mix.compute_AUC(fpr[1,:],tpr[1,:])
    auc[2] = auc2
    measure_all.append(auc)
    
    rtn = mix.test_RKDE(data_raw, m, n, kernel=kernel, norm=True, p_unre=p_unre1)
    out3, cluster_index, conf3, auc3, time3 = rtn  
    fpr = np.zeros((2,len(out1)))
    tpr = np.zeros((2,len(out1)))
    for i in range(len(out1)):
        n_cluster_pre, evl_measure = evaluation.evaluation_basic(out3[i], conf3[i], truth, truth_conf)
        #evl_measure = TP,FP,FN,TN, precision,recall,F1,FPR, MAE,RMSE
        if(evl_measure.shape[0]<2):
            evl_measure = np.append(evl_measure, np.zeros((1,10)),axis=0)
        fpr[:,i] = evl_measure[:,7]
        tpr[:,i] = evl_measure[:,5]
    auc = np.zeros(3)
    auc[0] = mix.compute_AUC(fpr[0,:],tpr[0,:])
    auc[1] = mix.compute_AUC(fpr[1,:],tpr[1,:])
    auc[2] = auc3
    measure_all.append(auc)

    if(save!=""):
        syn.save_syn(data_raw, truth, truth_conf, path=save)
    time1.append(size[0])
    time1.append(size[1])
    return([measure_all,time1[:]])

def run_test_mix(m=200, n=100, k=50):
    measure_all = []    
    lbda_set=[10,15,20,25,30]
    p_unre_set=[0.1,0.2,0.3]
    time_avg = []
    for lbda in lbda_set:
        for p_unre in p_unre_set:
            print "\n===lbda:"+str(lbda)+" p:"+str(p_unre)+"==="
            for ki in range(k):
                print "the "+str(ki+1)+"th dataset for lbda="+str(lbda)+" p="+str(p_unre)
                save = ""
                load = ".//data_syn//data_syn_mix"+"_"+str(lbda)+"_"+str(p_unre)+"_"+str(ki+1)+".txt"
                measure,time1 = run_syn_mix(m, n, load, save, lbda, lbda, p_unre, p_unre)
                measure_all.append(measure)
                time_avg.append(time1)
            avg_m = np.mean(np.array(measure_all),axis=0)
            np.savetxt(".//measure_syn//syn_auc_avg_mix"+"_"+str(lbda)+"_"+str(p_unre)+".txt",avg_m, fmt='%.4f')
    np.savetxt(".//measure_syn//time_avg_mix.txt",np.array(time_avg),fmt="%.4f")
    return([])
