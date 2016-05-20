# -*- coding: utf-8 -*-
"""
data_tripadvisor.py

@author: Mengting Wan
"""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as pt
import scipy.stats as sps
from scipy.stats import kde

name = ["Overall","Value","Rooms","Location",
        "Cleanliness","Check in/front desk","Service","Business service"]
colorc = "bgrcmykw"

def readfiles(path):
    fileHandle = open(path)
    f = fileHandle.readline()
    ratings = []
    users = []
    while(f!=''):
        dataline = f.strip()
        rating_usr = -np.ones(8)
        usr = ""
        while(dataline != ''):
            if dataline[:8]=="<Author>":
                usr = dataline[8:]
            if dataline[:9]=="<Overall>":
                rating_usr[0] = np.int64(dataline[9:])
            elif dataline[:7]=="<Value>":
                rating_usr[1] = np.int64(dataline[7:])
            elif dataline[:7]=="<Rooms>":
                rating_usr[2] = np.int64(dataline[7:])
            elif dataline[:10]=="<Location>":
                rating_usr[3] = np.int64(dataline[10:])
            elif dataline[:13]=="<Cleanliness>":
                rating_usr[4] = np.int64(dataline[13:])
            elif dataline[:23]=="<Check in / front desk>":
                rating_usr[5] = np.int64(dataline[23:])
            elif dataline[:9]=="<Service>":
                rating_usr[6] = np.int64(dataline[9:])
            elif dataline[:18]=="<Business service>":
                rating_usr[7] = np.int64(dataline[18:])
            f = fileHandle.readline()
            dataline = f.strip()
        f = fileHandle.readline()
        ratings.append(rating_usr)
        users.append(usr.lower())
    return([users,np.array(ratings)])

def plot_hist(item, figure_id=1):
    pt.figure(figure_id)
    kurtosis = -np.ones(8)
    for i in range(item.shape[1]):
        pt.subplot(240+i)
        tmp = item[item[:,i]!=-1,i]
        tmp = tmp + np.random.rand(len(tmp)) - 0.5
        pt.hist(tmp, bins=6, normed=True, range=(0.9,6.1), alpha=0.8, color=colorc[i])
        pt.title(name[i])
        density = kde.gaussian_kde(tmp)
        xgrid = np.linspace(0, 6, 100)
        pt.plot(xgrid, density(xgrid), 'r-')
        avg = np.mean(tmp)
        sd = np.std(tmp)
        pt.plot(xgrid, normpdf(xgrid,avg,sd))
        pt.show()
        kurtosis[i] = sps.kurtosis(item[item[:,i]!=-1,i])
    return(kurtosis)

def loadfiles(dir_path):
    files = os.listdir(dir_path)
    item_list = [[],[],[],[],[],[],[],[]]
    sources = {}
    sr_out = []
    for filenames in files:
        users,ratings = readfiles(dir_path + "//" + filenames)
        tmp = [[],[],[],[],[],[],[],[]]
        for i in range(len(users)):         
            if(users[i]!="a tripadvisor member" and users[i]!="lass="):
                if not sources.has_key(users[i]):
                    sources[users[i]] = len(sources)
                    sr_out.append(users[i])
                for j in range(8):
                    if ratings[i,j]>0:
                        tmp[j].append([sources[users[i]],ratings[i,j]])     
        for j in range(8):
            item = item_list[j][:]
            item.append(np.array(tmp[j]))
            item_list[j] = item[:]
    return([item_list, sr_out])


def Read():
    dir_path = ".//data_tripadvisor//Review_Texts"
    item_list, sources = loadfiles(dir_path)
    n_vec = np.zeros(8)
    for j in range(8):
        n_vec[j] = len(item_list[j])
    return([item_list, len(sources), n_vec, sources])

#data_list, m, n_vec, source = Read()
#item = item_list[2]
#plot_hist(item)