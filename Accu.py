# -*- coding: utf-8 -*-
"""
Accu.py

@author: Xiangyu Joe Chen; Mengting Wan
"""

from __future__ import division

import math
import numpy as np
import numpy.linalg as la
import basic_functions as bsf

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Algorithm FrameWork

Input: S, O.
Output: The true value for each object in O.

1. Set the accuracy of each source as 1 − errorrate;
2. while (accuracy of sources changes && no oscillation of decided true values):
    Compute probability of dependence between each pair of sources;
    Sort sources according to the dependencies;
    Compute confidence of each value for each object;
    Compute accuracy of each source;
3. for each (o ∈ O):
    Among all values of O, select the one with the highest confidence as the true value;
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Extract data
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def extract(data, m, n):
    index=[]
    claim=[]
    src_dict={}
    for i in range(n):
        src = list(data[i][:,0])
        for j in src:
            if(src_dict.has_key(j)):
                tmp = src_dict[j][:]
                tmp.append(i)
                src_dict[j] = tmp[:]
            else:
                src_dict[j] = [i]
        index.append(src)
        claim.append(data[i][:,1])
    return([index,claim,src_dict])

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Initialization of dependence, confidence, and accuracy array; as well as the truth
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def initialization(alpha, c, err, m, n, claim):
    dependence = np.zeros( (m, m) )
    confidence = np.zeros( (n, m) )
    accuracy = np.zeros(m)
    truth = []

    for i in range(m):
        accuracy[i] = 1-err

    for i in range(n):
        truth.append( claim[i][0] )

    return dependence, confidence, accuracy, truth


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Evaluate Dependence between every pair of srcs
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def evalDependence(alpha, index, claim, accuracy, dependence, m, n, nn, c, truthList, src_dict):
    # claim for each src
    srcClaim = [[None for _ in range(n)] for _ in range(m)]

    # generate claim list for each src
    for i in range(m):
        for j in range(n):
            if i in index[j]:
                srcClaim[i][j] = claim[j][index[j].index(i)]

    #print 'srcClaim:'
    #print "srcClaim ====================================="
    #print  srcClaim
    #print "==============================================\n"

    # evaluate dependence between each src pair
    mark = {}
    for i in range(m):
        srcCand = np.array([])
        if(src_dict.has_key(i)):
            for j in src_dict[i]:
                srcCand = np.append(srcCand,index[j])
            srcCand = list(set(srcCand))
            for j in srcCand:
                j = int(j)
                if not mark.has_key((i,j)):
                    mark[(i,j)] = 1
                    mark[(j,i)] = 1
                    numTrue = 0.0
                    numFalse = 0.0
                    numDiff = 0.0
                    
                    entity_inter = list(set(np.append(src_dict[i],src_dict[j])))
                    
                    if(len(entity_inter)>0):
                        for k in entity_inter:
                            if srcClaim[i][k] is not None and srcClaim[j][k] is not None:
                                if srcClaim[i][k] == srcClaim[j][k] and srcClaim[i][k] == truthList[k]:
                                    numTrue = numTrue+1
            
                                if srcClaim[i][k] == srcClaim[j][k] and srcClaim[i][k] != truthList[k]:
                                    numFalse = numFalse+1
            
                                if srcClaim[i][k] != srcClaim[j][k]:
                                    numDiff = numDiff+1
            
                        independProbCond = ( (accuracy[i]*accuracy[j])**numTrue*((1-accuracy[i])*(1-accuracy[j]))**numFalse*(1-(accuracy[i]*accuracy[j])-(1-accuracy[i])*(1-accuracy[j])/nn)**numDiff )/(nn**numFalse)
                        dependProbCond1 = (accuracy[i]*c+accuracy[i]*accuracy[j]*(1-c))**numTrue*((1-accuracy[i])*c+(1-accuracy[i])*(1-accuracy[j])/nn*(1-c))**numFalse*(1-(accuracy[i]*accuracy[j])-(1-accuracy[i])*(1-accuracy[j])/nn*(1-c))**numDiff
                        dependProbCond2 = (accuracy[j]*c+accuracy[i]*accuracy[j]*(1-c))**numTrue*((1-accuracy[j])*c+(1-accuracy[i])*(1-accuracy[j])/nn*(1-c))**numFalse*(1-(accuracy[i]*accuracy[j])-(1-accuracy[i])*(1-accuracy[j])/nn*(1-c))**numDiff
                        dependProbCond = dependProbCond1+dependProbCond2
            
                        #print i, j
                        #print independProbCond, dependProbCond
            
                        dependProb = (dependProbCond*alpha)/(dependProbCond*alpha+independProbCond*(1-alpha))
                        #print dependProb
            
                        dependence[i][j] = dependProb
                        dependence[j][i] = dependProb

    return dependence


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Evaluate Confidence for each object value set
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def evalConfidence(c, index, claim, dependence, confidence, accuracy, m, n, sim=True, rho=0.9):
    # evaluate accuracy score, namely A'(S)
    accuScore = np.zeros(m)

    for i in range(m):
        if(accuracy[i]>0 and accuracy[i]<1):
            accuScore[i] = np.log( n*accuracy[i]/(1-accuracy[i]) )
        elif(accuracy[i]>0):
            accuScore[i] = 1e10


    # go through claims on each object iteratively, estimate confidence accordingly
    for i in range(n):
        for j in range( len(claim[i]) ):
            value = claim[i][j]
            confValue = 0
            srcList = []
            #print i, j, value

            for k in range(len(claim[i])):
                if claim[i][k] == value:
                    srcList.append( index[i][k] )
         
            #print srcList
            # sort srcList based on dependence
            srcListSorted = []

            for l in range( len(srcList) ):
                dependValue = -1
                for o in range( len(srcList) ):
                    if dependence[l][o] > dependValue:
                        dependValue = dependence[l][o]

                srcListSorted.append( [srcList[l], dependValue] )

            #print srcListSorted
            # evaluate I(S)
            preS = []
            for l in range( len(srcListSorted) ):
                iS = 1
                for o in range( len(preS) ):
                    '''
                    print 'lalala'
                    print srcListSorted[l][0], preS[o]
                    print dependence[2][0]
                    print dependence[ int(srcListSorted[l][0]) ][ int(preS[o]) ]
                    '''
                    iS = iS*(1-c*dependence[ int(srcListSorted[l][0]) ][ int(preS[o]) ])

                preS.append( srcListSorted[l][0] )
                confValue = confValue+iS*accuScore[ int(srcListSorted[l][0]) ]
                #print confValue

            confidence[i][index[i][j]] = confValue
            #print confValue
        if(sim):
            tmp = np.copy(confidence[i])
            for j in range( len(claim[i]) ):
                tmp[index[i][j]] = (1-rho)*confidence[i][index[i][j]] + rho*sum(np.exp(-abs(claim[i]-claim[i][j]))*confidence[i][index[i]])
            confidence[i] = np.copy(tmp)
    return confidence


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Evaluate accuracy for each src
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def evalAccuracy(index, claim, confidence, accuracy, m, n, src_dict):
    # evaluate P(v)
    pV = []

    for i in range(n):
        uniConfidence = []
        omega = 0.0
        pVObject = []

        for j in range( len(claim[i]) ):
            if [claim[i][j], confidence[i][index[i][j]]] not in uniConfidence:
                uniConfidence.append( [claim[i][j], confidence[i][index[i][j]]] )

        #print uniConfidence
        #print '\n'

        for j in range( len(uniConfidence) ):
            omega = omega+math.exp( uniConfidence[j][1]-confidence[i].max() )
            #print uniConfidence[k][1]
            #print omega
        #print '////////////'

        for j in range( len(uniConfidence) ):
            pVObject.append( [ math.exp(uniConfidence[j][1]-confidence[i].max())/omega, uniConfidence[j][0] ] )

        pV.append(pVObject)

    #print pV

    # evaluate A(S)
    for i in range(m):
        aS = 0
        aSList = []
        
        if(src_dict.has_key(i)):
    
            for j in src_dict[i]: 
                value = None
    
                if i in index[j]:
                    #print index[j].index( str(i) )
                    value = claim[j][ index[j].index(i) ]
    
                for k in range( len(pV[j]) ):
                    if value in pV[j][k]:
                        aSList.append( pV[j][k][0] )
    
            #print aSList
            if(len(aSList)>0):
                for j in range( len(aSList) ):
                    aS = aS+aSList[j]
                aS = aS/len(aSList)
    
            accuracy[i] = aS

    #print accuracy
    return accuracy


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Main Algorithm of AccuSim/AccuCopy
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def AccuSim(data, m, n, alpha = 0.2, c = 0.8, err = 0.2, nn = 10):
    index, claim, src_dict = extract(data, m, n) 
    truthList = []
    itr = 0
    #print "index claim count ============================"
    #print index, claim, count
    #print "==============================================\n"

    dependence, confidence, accuracy, truthList = initialization(alpha, c, err, m, n, claim)
    
    while(itr < 15 and err>0.1):
        tmp = np.copy(truthList)
        dependence = evalDependence(alpha, index, claim, accuracy, dependence, m, n, nn, c, truthList, src_dict)
        confidence = evalConfidence(c, index, claim, dependence, confidence, accuracy, m, n)
        accuracy = evalAccuracy(index, claim, confidence, accuracy, m, n, src_dict)

        #find truth: values with max confidence
        truthList = []

        for i in range(n):
            truthObject = -1
            claim_tmp = -np.ones(m)
            claim_tmp[index[i]] = claim[i]
            truthObject = claim_tmp[ confidence[i].argmax()]
            truthList.append(truthObject)

        print "iteration " + str(itr)
        #print confidence
        #print accuracy
        #print truthList
        #print "======================================\n"
        itr = itr+1
        err = la.norm(tmp-np.copy(truthList))/la.norm(tmp)
        print err

    return(np.copy(truthList))


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Data used for testing purpose
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
if __name__=="__main__":
    # Example in paper, for testing purpose
    # number of sources
    m = 5
    # number of entities
    n = 5
    # ni*2 array
    data = [ np.array([ ['0','MIT'],      ['1','Berkeley'],  ['2','MIT'],      ['3','MIT'],     ['4','MS'] ]),
             np.array([ ['0','MSR'],      ['1','MSR'],       ['2','UWise'],    ['3','UWise'],   ['4','UWise'] ]),
             np.array([ ['0','MSR'],      ['1','MSR'],       ['2','MSR'],      ['3','MSR'],     ['4','MSR'] ]),
             np.array([ ['0','UCI'],      ['1','AT&T'],      ['2','BEA'],      ['3','BEA'],     ['4','BEA'] ]),
             np.array([ ['0','Google'],   ['1','Google'],    ['2','UW'],       ['3','UW'],      ['4','UW'] ]) ]
    data_array = np.array(data)

    accuCopy(data, m, n)
'''

alpha = 0.2
c = 0.8
err = 0.2
nn = 10