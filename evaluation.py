# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:23:44 2015

@author: Mandy
"""
from __future__ import division

import numpy as np
import basic_functions as bsf

def refine_single(truth,conf):
    truth_single = []
    n = len(truth)
    for i in range(n):
        truth_tmp = np.sort(truth[i])
        conf_tmp = conf[i][np.argsort(truth[i])]
        if(len(truth_tmp)>0):
            truth_single.append(truth_tmp[np.argmax(conf_tmp)])
        else:
            truth_single.append(np.array([]))
    return(np.array(truth_single))

def refine_single_m(truth_m, conf_m):
    k = len(truth_m)
    truth_single_m = np.zeros((len(truth_m[0]),k))
    for i in range(k):
        truth_single_m[:,i] = refine_single(truth_m[i][:],conf_m[i][:])
    return(truth_single_m)


def evaluation_basic(out, out_conf=[], truth=[], truth_conf=[]):
    if(len(truth)==0 and len(truth_conf)==0):
        if(len(out_conf)>0):
        ##### Case 1 #####
        # evaluate output and its consistency without knowing groundtruth
            n = len(out)
            n_cluster = np.zeros(n)
            for i in range(n):
                n_cluster[i] = len(out[i])
            #print "#objects:", n
            #print "#objects who may have more than one cluster:", sum(n_cluster>1)
            return([n_cluster])
    if(len(truth)>0 and len(truth_conf)==0):
    ##### Case 2 #####
    # evaluate output and its consistency with knowing groundtruth, where goundtruth is consistent
        n = len(out)
        print "#objects:", n        
        if(len(out_conf)>0):
            n_cluster = np.zeros(n)
            for i in range(n):
                n_cluster[i] = len(out[i])
            #print "#objects who may have more than one cluster:", sum(n_cluster>1)
            out_single = refine_single(out,out_conf)
            MAE = np.mean(abs(out_single-truth))
            RMSE = np.sqrt(np.mean((out_single-truth)**2))
            #print "MAE=", MAE
            #print "RMSE=", RMSE
            #print "Err=", Err
            return([n_cluster, MAE, RMSE])
        else:
            out_single = np.copy(out)
            #print "#objects who may have more than one cluster:", 0
            MAE = np.mean(abs(out_single-truth))
            RMSE = np.sqrt(np.mean((out_single-truth)**2))
            #print "MAE=", MAE
            #print "RMSE=", RMSE
            #print "Err=", Err
            return([MAE, RMSE])
    if(len(truth)>0 and len(truth_conf)>0 and len(out_conf)>0):
    ##### Case 3 #####
    # evaluate output and its consistency with knowing groundtruth, where goundtruth is controversial
        n = len(out)
        #print "#objects:", n
        n_cluster_tr = np.zeros(n)
        n_cluster_pre = np.zeros(n)
        for i in range(n):
            n_cluster_tr[i] = len(truth[i])
            n_cluster_pre[i] = len(out[i])
        k = max(n_cluster_tr)
        evl_measure = []
        for i in range(np.int64(k)):
            ind_tr = np.where(n_cluster_tr==i+1)[0]
            ind_pre = np.where(n_cluster_pre==i+1)[0]
            ind_tp = list(set(ind_tr) & set(ind_pre))
            TP = len(ind_tp)
            FP = len(ind_pre) - TP
            FN = len(ind_tr) - TP
            TN = n - len(ind_tr) - FP
            if(TP+FP>0):
                precision = TP/np.float(TP+FP)
            else:
                precision = 0
            recall = TP/np.float(TP+FN)
            if(FP+TN>0):
                FPR = FP/(FP+TN)
            else:
                FPR = 0
            F1 = 2*TP/(2*TP+FP+FN)
            MAE = 0
            RMSE = 0
            for j in range(TP):
                tmp1 = np.array(sorted(out[ind_tp[j]]))
                tmp2 = np.array(sorted(truth[ind_tp[j]]))
                MAE = MAE + sum(abs(tmp1-tmp2))
                RMSE = RMSE + sum((tmp1-tmp2)**2)
            if(TP>0):
                MAE = MAE/(TP*(i+1))
                RMSE = np.sqrt(RMSE/(TP*(i+1)))
            measure = [TP,FP,FN,TN, precision,recall,F1,FPR, MAE,RMSE]
            evl_measure.append(measure)
        return([n_cluster_pre, np.array(evl_measure)]) 
    return(0)
            
def compare_single(out_m, truth, n_mark=-1, conf_m=[], prt=False):
    if(len(conf_m)>0):
        out_single0 = refine_single_m(out_m, conf_m)
        k = len(out_m)
    else:
        out_single0 = np.copy(out_m)
        k = len(out_m[0,:])    
    MAE = np.zeros(k)
    RMSE = np.zeros(k)
    if(n_mark>0):
        out_single = out_single0[:n_mark,:]
    else:
        out_single = np.copy(out_single0)
    for i in range(k):
        MAE[i] = np.mean(abs(out_single[:,i]-truth))
        RMSE[i] = np.sqrt(np.mean((out_single[:,i]-truth)**2))
    if(prt):
        print "MAE=",MAE
        print "RMSE=",RMSE
    return([out_single,np.array([MAE,RMSE])])


def evaluation_source_single(data, m, n, truth):
    index, claim, count = bsf.extract(data, m, n)
    mae = np.zeros(m)
    rmse = np.zeros(m)
    for i in range(n):
        mae[index[i]] = mae[index[i]] + abs(claim[i]-truth[i])
        rmse[index[i]] = rmse[index[i]] + (claim[i]-truth[i])**2
    mae[count>0] = mae[count>0]/count[count>0]
    rmse[count>0] = np.sqrt(rmse[count>0]/count[count>0])
    rtn = np.append(mae.reshape(m,1),rmse.reshape(m,1),axis=1)
    rtn = np.append(rtn,count.reshape(m,1),axis=1)
    return(rtn)
    
def evaluation_source_multiple(data, m, n, truth):
    index, claim, count = bsf.extract(data, m, n)
    mae = np.zeros(m)
    rmse = np.zeros(m)
    for i in range(n):
        cluster_ind = data[i][:,2]
        ind = np.array(index[i])
        index_new = list(ind[cluster_ind>=0])
        index_noise = list(ind[cluster_ind<0])
        tmp = truth[i][list(cluster_ind[cluster_ind>=0])]
        mae[index_new] = mae[index_new] + abs(claim[i]-tmp)
        rmse[index_new] = rmse[index_new] + (claim[i]-tmp)**2
        count[index_noise] = -1
    mae[count>0] = mae[count>0]/count[count>0]
    rmse[count>0] = np.sqrt(rmse[count>0]/count[count>0])
    return([mae, rmse, count])
