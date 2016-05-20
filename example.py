# -*- coding: utf-8 -*-
"""
test_real.py

@author: Mengting Wan
"""

from __future__ import division

import numpy as np
import evaluation
import test_functions as tf

def data():
    x1 = np.array([[0, 1.0],
                   [1, 1.1],
                   [2, 0.9],
                   [4, 5.9]])
    x2 = np.array([[0, 3.0],
                   [1, 3.1],
                   [2, -3.0],
                   [3, -3.1],
                   [4, 5.0],
                   [5, -2.9],
                   [6, -3.05]])
    x3 = np.array([[0, 1.0],
                   [1, 0.9],
                   [3, 1.1],
                   [4, -5.0]])
    return([[x1,x2,x3], 7, 3])

def run():
    data_raw, m, n = data()
    src_score = []
    for i in range(1,8):
        rtn = tf.test_KDEm(data_raw, m, n, kernel="gaussian", norm=True, max_itr=i)
        out1, cluster_index, conf1, source_score, weights_for_each, moments1, time1 = rtn
        src_score.append(source_score)
    print src_score
    np.savetxt("src_score_example.txt", np.array(src_score))

if __name__=="__main__":
    run()
