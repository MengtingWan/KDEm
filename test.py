# -*- coding: utf-8 -*-
"""
test.py

@author: Mengting Wan
"""
import sys

import test_real as real
import test_syn_uni as syn_uni
import test_syn_mix as syn_mix

if __name__=="__main__":
    print "================================="
    #print "Definition: test -data -method"
    print "Definition: test -data"
    print "Parameters"
    print "    data:    the dataset indicater"
    print "             should be one of 'synuni', 'synmix', 'realpop' and 'realtrip'"
    print "             'synuni': run experiments on Synthetic(unimodal) datasets"
    print "             'synmix': run experiments on Synthetic(mix) datasets"
    print "             'realpop': run experiments on the Population(outlier) dataset"
    print "             'realtrip': run experiments on Tripadvisor datasets"
    print "             the default is 'realpop'."
    #print "    method:  the method indicater"
    #print "             should be one from 'kernel', 'all'"
    #print "             'kernel': run only KDEm/KDE on indicated datasets"
    #print "             'all': run KDEm/KDE, and all other baseline methods on indicated datsets"
    #print "             the default is 'kernel'"
    #print "             Notice that results from 'kernel' and 'all' are the same on Synthetic(mix) datasets and Tripadvisor datasets"
    print "Example:     test -realpop"
    print "=================================="
    if(len(sys.argv)>1):
        arg = sys.argv[0]
        if arg=="synuni":
            syn_uni.run_syn_uni(m=200, n=100, k=50)
        elif arg=="synmix":
            syn_mix.run_syn_mix(m=200, n=100, k=50)
        elif arg=="realpop":
            real.run_test_pop()
        elif arg=="realtrip":
            real.run_test_tripadvisor()
        else:
            "Error: the input parameter should be one of 'synuni', 'synmix', 'realpop' and 'realtrip'"
    else:
        real.run_syn_uni(m=200, n=100, k=50)
                
            