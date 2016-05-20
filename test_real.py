# -*- coding: utf-8 -*-
"""
test_real.py

@author: Mengting Wan
"""

from __future__ import division

import numpy as np
import data_pop as pop
import data_tripadvisor as tripadvisor
import evaluation
import test_functions as tf


def SaveVecList(list_ary, path):
    fil = open(path,"w")
    for k in range(len(list_ary)):
        for i in range(len(list_ary[k])):
            print>>fil,"%.4f"%list_ary[k][i],'\t',
        print>>fil
    fil.close()

def run_test_pop():
    data_raw, m, n, objects, sources, ground_truth, n_mark = pop.Read()
    src_r_truth = evaluation.evaluation_source_single(data_raw, m, n_mark, ground_truth)

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
                                                    ground_truth, n_mark, [conf1,conf2,conf2r,conf3,conf4,conf4r])
    
    #Test with baseline methods and report test results
    out_other = tf.test_basic(data_raw, m, n, tp="maximal")
    out_single_bs, measure_bs = evaluation.compare_single(out_other, ground_truth, n_mark)
    
    #Save estimated source reliability scores and true source reliability scores based on MAE and RMSE respectively
    np.savetxt("pop_src_score.txt", np.append(source_score.reshape((m,1)),src_r_truth, axis=1), fmt="%.4f")
    
    #Save test results
    measure_all = np.append(measure, measure_bs, axis=1)
    np.savetxt("pop_measure.txt", measure_all.transpose(), fmt='%.0f')
    
    #Save outputs
    out_all = np.append(out_single,out_single_bs,axis=1)
    np.savetxt("pop_output.txt", np.append(out_all,ground_truth.reshape(n_mark,1),axis=1), fmt='%.4f')        
    return()

def run_test_tripadvisor():
    data_list, m, n_vec, source = tripadvisor.Read()
    thr = 0.2
    src_score = []
    ni = []
    for j in range(len(data_list)):        
        #Test with KDEm
        print "the "+str(j+1)+"th dataset in Tripadvisor"
        rtn = tf.test_KDEm_fast(data_list[j][:], m, int(n_vec[j]), 
                                kernel="gaussian", norm=False, outlier_thr=thr, h=0.8)
        out, cluster_index, cluster_conf, source_score, weights_for_each, moments, time0 = rtn
        
        src_score.append(source_score)          
        ni.append(evaluation.evaluation_basic(out, cluster_conf,[],[])[0])
        SaveVecList(out, "tripadvisor_output"+str(j+1)+".txt")
        SaveVecList(out, "tripadvisor_conf"+str(j+1)+".txt")
        np.savetxt("tripadvisor_moments"+str(j+1)+".txt", moments, fmt='%.4f')
        output_sel(data_list[j], ni[j], 1, 2, "uni-ex-tr-claims-"+str(j+1)+".txt", "uni-ex-tr-src-"+str(j+1)+".txt")
        output_sel(data_list[j], ni[j], 2, 2, "bi-ex-tr-claims-"+str(j+1)+".txt", "bi-ex-tr-src-"+str(j+1)+".txt")
        output_sel(data_list[j], ni[j], 3, 99, "tri-ex-tr-claims-"+str(j+1)+".txt", "tri-ex-tr-src-"+str(j+1)+".txt")
    np.savetxt("tripadvisor_src_score.txt", np.array(src_score).transpose(), fmt='%.4f') 
    np.savetxt("tripadvisor_ni.txt", np.array(ni), fmt="%.4f")
    return([])

def output_sel(data_raw, ni, thr=3, cut=99, path_claim="tri-ex-claims.txt", path_src="tri-ex-sources.txt"):
    ind = np.where(ni==thr)[0]
    file0 = open(path_claim,"w")
    file1 = open(path_src,"w")
    for i in range(min(len(ind),cut)):
        tmp = np.copy(data_raw[ind[i]])
        for j in range(tmp.shape[0]-1):
            print>>file0, tmp[j,1],
            print>>file1, tmp[j,0],
        print>>file0, tmp[j+1,1]
        print>>file1, tmp[j+1,0]
    file0.close()
    file1.close()
    

'''
if __name__=="__main__":
    run_test_pop()
    run_test_tripadvisor()
'''