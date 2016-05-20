# -*- coding: utf-8 -*-
"""
test_syn.py

@author: Mengting Wan
"""


from __future__ import division

import numpy as np
import data_syn as syn
import evaluation
import test_functions as tf

def save_list_array(list_ary, path, header=''):
    fil = open(path,"w")
    if(header!=''):
        print>>fil,header
    for k in range(len(list_ary)):
        for i in range(list_ary[k].shape[0]):
            for j in range(list_ary[k].shape[1]-1):
                print>>fil, "%.4f"%list_ary[k][i,j],'\t',
            print>>fil, "%.4f"%list_ary[k][i,j+1]
        print>>fil
    fil.close()

def run_syn_uni(m=200, n=100, load="", save="", lbda=5, p_unre=0):
    if(load!=""):
        data_raw, truth, truth_conf = syn.reload_data(load)
    else:
        data_raw, truth, truth_conf = syn.data_unimodal(m, n, lbda, p_unre)
    size = np.zeros(2)
    for i in range(n):
        size = size + np.array([data_raw[i].shape[0],data_raw[i].shape[0]**2])
    
    #Test with KDEm
    rtn = tf.test_KDEm(data_raw, m, n, kernel="gaussian", norm=True)
    out1, cluster_index, conf1, source_score, weights_for_each, moments1, time1 = rtn                
    
    #Test with KDE
    rtn = tf.test_KDE(data_raw, m, n, kernel="gaussian", norm=True)
    out2, cluster_index, conf2, moments2, time2 = rtn
    
    #Test with RKDE
    rtn = tf.test_RKDE(data_raw, m, n, kernel="gaussian", norm=True, argmax=False)
    out2r, cluster_index, conf2r, weights_for_each, moments, time0  = rtn    
    
    #Test with KDEm
    rtn = tf.test_KDEm(data_raw, m, n, kernel="gaussian", norm=True, argmax=True)
    out3, cluster_index, conf3, source_score, weights_for_each, moments1, time1 = rtn                
    
    #Test with KDE
    rtn = tf.test_KDE(data_raw, m, n, kernel="gaussian", norm=True, argmax=True)
    out4, cluster_index, conf4, moments2, time2 = rtn
    
    #Test with RKDE
    rtn = tf.test_RKDE(data_raw, m, n, kernel="gaussian", norm=True, argmax=True)
    out4r, cluster_index, conf4r, weights_for_each, moments, time0  = rtn
    
    #Select the most significant representative value and report test results for KDEm and KDE
    out_single, measure = evaluation.compare_single([out1,out2,out2r,out3,out4,out4r], 
                                                    np.array(truth)[:,0], n, [conf1,conf2,conf2r,conf3,conf4,conf4r])
                                                    
    #Test with baseline methods and report test results
    out_other = tf.test_basic(data_raw, m, n, tp="maximal")
    out_single_bs, measure_bs = evaluation.compare_single(out_other, np.array(truth)[:,0], n, prt=False)
    
    measure_all = np.append(measure, measure_bs, axis=1)
    #np.savetxt("syn_uni_measure.txt", measure_all, fmt='%.4f')
    
    #Save data
    if(save!=""):
        syn.save_syn(data_raw, truth, truth_conf, path=save)
    
    time1.append(size[0])
    time1.append(size[1])
    return([measure_all,time1])

def run_test_uni(m=200, n=100, k=50):
    lbda_set=[3,5,7,9]
    p_unre_set=[0.1,0.2,0.3]
    time_avg = []
    for lbda in lbda_set:
        for p_unre in p_unre_set:
            measure_all = []
            print "\n===lbda:"+str(lbda)+" p:"+str(p_unre)+"==="
            for ki in range(k):
                print "the "+str(ki+1)+"th dataset for lbda="+str(lbda)+" p="+str(p_unre)
                save = ""
                load = ".//data_syn//data_syn_uni"+"_"+str(lbda)+"_"+str(p_unre)+"_"+str(ki+1)+".txt"
                measure,time1 = run_syn_uni(m, n, load, save, lbda, p_unre)
                measure_all.append(np.array(measure[:]))
                time_avg.append(time1)
            avg_m = np.mean(measure_all,axis=0)
            #std_m = np.std(measure_all,axis=0)/np.sqrt(k)
            np.savetxt(".//measure_syn//syn_measure_avg_uni"+"_"+str(lbda)+"_"+str(p_unre)+".txt", avg_m.transpose(),fmt="%.4f")
            #np.savetxt(".//measure_syn//syn_measure_std_uni"+"_"+str(lbda)+"_"+str(p_unre)+".txt", std_m.transpose(),fmt="%.4f")
    np.savetxt(".//measure_syn//time_avg_uni.txt",np.array(time_avg),fmt="%.4f")
    return([])
