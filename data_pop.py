# -*- coding: utf-8 -*-
"""
data_pop.py

@author: Mengting Wan
"""

from __future__ import division

import numpy as np

def check_repeat(recd):
    sr_recd = {}
    sr_recd[recd[0][0]] = 0
    recd_new = [recd[0][:]]
    l = len(recd)
    for i in range(1,l):
        if(sr_recd.has_key(recd[i][0])):
            recd_new[sr_recd[recd[i][0]]][1] = np.copy(recd[i][1])
        else:
            sr_recd[recd[i][0]]=len(sr_recd)
            recd_new.append(recd[i][:])
    return(recd_new)

def ReadData(ground_truth):
    fileHandle = open(".//data_pop//popTuples.txt")
    f = fileHandle.readline()
    data_mark = []
    data_unmark = []
    truth_new = []
    
    objects_mark = []
    objects_unmark = []
    data_obj = {}
    
    while(f!=''):
        dataline = f.strip()
        item = dataline.split("\t")
        ob = item[0].lower()
        year = item[6][5:9]
        ob = ob+":"+year
        sr = item[4].lower()
        tmp = np.int(item[7])    
        if(tmp<1e8):
            if(data_obj.has_key(ob)):
                recd = data_obj[ob][:]
                recd.append([sr, tmp])
                data_obj[ob] = recd[:]
            else:    
                recd = [[sr, tmp]]
                data_obj[ob] = recd[:]
        f=fileHandle.readline()
    fileHandle.close()
    
    sources = {}
    sr_out = []
    for (k,v) in data_obj.items():
        v = check_repeat(v)[:]
        n_v = len(v)
        srs = np.zeros(n_v)
        vls = np.zeros(n_v)        
        for j in range(n_v):
            vls[j] = v[j][1]
        if(np.var(vls)>0):
            for j in range(n_v):
                if(sources.has_key(v[j][0])):
                    srs[j] = sources[v[j][0]]
                else:
                    sources[v[j][0]] = len(sources)
                    sr_out.append(v[j][0])
                    srs[j] = sources[v[j][0]]
            if(ground_truth.has_key(k)):
                data_mark.append(np.array([srs,vls]).transpose())
                truth_new.append(ground_truth[k])
                objects_mark.append(k)
            else:
                data_unmark.append(np.array([srs,vls]).transpose())
                objects_unmark.append(k) 
    objects = np.append(objects_mark[:], objects_unmark[:])
    n_mark = len(data_mark)
    n_unmark = len(data_unmark)
    for i in range(n_unmark):
        data_mark.append(data_unmark[i])        
    return([data_mark, objects, sr_out, np.array(truth_new), n_mark])

def ReadTruth():
    fileHandle = open(".//data_pop//popAnswersOut.txt")
    f = fileHandle.readline()
    truth = {}
    while(f!=''):
        dataline = f.strip()
        item = dataline.split(',')
        ob = item[0]+','+item[1]+':'+item[2].strip()
        tr = np.int64(item[3])
        #truth[ob] = np.log(tr+1)
        truth[ob] = tr
        f = fileHandle.readline()
    fileHandle.close()
    return(truth)

   
def Read():
    ground_truth = ReadTruth()
    data_raw, objects, sources, truth, n_mark = ReadData(ground_truth)
    m = len(sources)
    n = len(data_raw)
    return([data_raw, m, n, objects, sources, truth, n_mark])